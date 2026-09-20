// =============================================================================
// ClTuneGemm.cpp -- KTT GEMM example driven by the CI tuning-budget library.
//
// Drop-in replacement for KTT's Examples/ClTuneGemm/ClTuneGemm.cpp.
//
// CHANGES vs the stock example:
//   1. tuner.Tune() is replaced by a loop of tuner.TuneIteration() calls --
//      each call benchmarks ONE configuration, and the loop performs tuning
//      only for as long as the CI tuner library says it is worth it.
//   2. The user specifies how many times CITuneGemm runs (X, command-line
//      argument 5). Tuning steps AND production GEMM executions both draw
//      from this budget: the library decides the split.
//   3. Per tuning step the measured KTT overhead (result.GetKernelOverhead())
//      is folded into the per-step cost pushed to the library, so the stop
//      decision uses MEASURED overhead, not an assumed constant.
//      NOTE: the CUDA backend leaves GetKernelOverhead() at 0, so the actual
//      per-step cost is taken as WALL-CLOCK around TuneIteration() (kernel +
//      NVRTC compilation + searcher + host overhead); overhead = wall - kernel.
//   4. Historical data is precomputed offline by Python
//      (src/precompute_historical.py -> historical_params.csv) and loaded by
//      the library's historical_cache at initiate_kernel() time -- nothing is
//      recomputed here. Point TUNER_HISTORICAL_PARAMS at the CSV or place it
//      next to the binary. Historical CSV data itself must be reachable at
//      raw-data/raw-autotuning-data/... relative to the CWD.
//
// BUILD (from the KTT build tree that already produced libktt):
//   g++ -O2 -std=c++17 -I<KTT>/include -I<Diplomka>/src \
//       ClTuneGemm.cpp <Diplomka>/src/tuner_api.c <Diplomka>/src/evaluator.c \
//       <Diplomka>/src/curvefit.c <Diplomka>/src/csv.c \
//       <Diplomka>/src/historical_cache.c <Diplomka>/src/minicsv.c \
//       -o ClTuneGemm -lKtt -lgsl -lgslcblas -llbfgs -lm -fopenmp
//
// USAGE:
//   ./ClTuneGemm [platform] [device] [kernelFile] [referenceKernelFile] \
//                <X: #executions> [mode: live|hybrid|hist] [histHW]
//   X is REQUIRED. Example: ./ClTuneGemm 0 0 ../Examples/ClTuneGemm/ClTuneGemm.cl \
//       ../Examples/ClTuneGemm/ClTuneGemmReference.cl 10000 hybrid 680
// =============================================================================

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#include "tuner_api.h"
#include <fstream>
#include <limits>
#include <chrono>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/ClTuneGemm/ClTuneGemm.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/ClTuneGemm/ClTuneGemmReference.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/ClTuneGemm/ClTuneGemm.cl";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/ClTuneGemm/ClTuneGemmReference.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = true;

// Toggle kernel profiling.
const bool useProfiling = false;

// Reduced tuning parameters set, taken from CLTune.
const bool useReducedSet = true;

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b)
{
    return ((a / b) * b == a) ? true : false;
};

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;
    std::string referenceKernelFile = defaultReferenceKernelFile;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));

            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);

                if (argc >= 5)
                {
                    referenceKernelFile = std::string(argv[4]);
                }
            }
        }
    }

    // ---- CI-tuner arguments -------------------------------------------------
    // X: how many times the user wants to run CITuneGemm. REQUIRED. Tuning
    // steps and post-tuning production executions both draw from this budget;
    // the library decides how many of the X executions are tuning steps.
    if (argc < 6)
    {
        std::cerr << "Missing required argument: <X: number of CITuneGemm executions>\n";
        std::cerr << "Usage: " << argv[0] << " [platform] [device] [kernelFile] [referenceKernelFile]"
                  << " <X> [mode: live|hybrid|hist] [histHW] [searcher: det|rand, default rand] [fast] [overhead_us]\n";
        return 1;
    }
    const uint64_t totalRuns = std::stoull(std::string(argv[5]));
    if (totalRuns == 0)
    {
        std::cerr << "X must be > 0\n";
        return 1;
    }
    const std::string modeStr = (argc >= 7) ? argv[6] : "live";
    const std::string histHW = (argc >= 8) ? argv[7] : "680";
    // Searcher selection: "rand" (DEFAULT) = RandomSearcher, deployment-style
    // random sampling with replacement -- what vanilla KTT/CLTune use and what
    // the harness simulations assume; every session is a fresh sample.
    // "det" = DeterministicSearcher, fixed lexicographic order -- the analysis
    // instrument for paired-stream comparisons (cross_engine prefix property);
    // NOT reproducible-session-relevant since rand streams are not reproducible.
    const bool useRandomSearcher = !(argc >= 9 && std::string(argv[8]) == "det");
    // "fast" (argv[9]): stop the program right after the tuner's stop
    // decision -- production runs convey no information about the DECISION
    // (session cost is computable: tuning sum + (X-S)*best), so skipping
    // them turns a ~50-minute experiment into a ~3-minute one. Use for
    // multi-seed testing of the stop decision itself.
    const bool fastMode = (argc >= 10 && std::string(argv[9]) == "fast");
    // Overhead (argv[10], us): feeds the HISTORICAL fit at init (history_run's
    // oracle scan prices each draw at kernel+overhead). The LIVE cost model is
    // unaffected -- it uses the per-step measured totals from push_result.
    // Default 0 = old behavior (free historical steps -> O_hist too deep).
    const uint64_t cfgOverhead = (argc >= 11) ? std::stoull(std::string(argv[10])) : 0;
    // Output files carry the full input signature, so runs never overwrite:
    //   tuning_steps_<X>_<mode>_hist<hw>_<searcher>[_oh<overhead>][_fast].csv
    std::string tag = std::to_string(totalRuns) + "_" + modeStr + "_hist" + histHW
                    + "_" + (useRandomSearcher ? "rand" : "det");
    if (fastMode) tag += "_fast";
    if (cfgOverhead > 0) tag += "_oh" + std::to_string(cfgOverhead);

    TunerMode tunerMode = TUNER_MODE_LIVE;
    double k = 1.0;
    if (modeStr == "hybrid")
    {
        tunerMode = TUNER_MODE_HYBRID;
        k = 0.5;
    }
    else if (modeStr == "hist")
    {
        tunerMode = TUNER_MODE_HISTORICAL;
        k = 0.0;
    }
    else if (modeStr != "live" && modeStr != "ref")
    {
        std::cerr << "Unknown mode '" << modeStr << "' (use live|hybrid|hist|ref)\n";
        return 1;
    }

    uint32_t kSizeM;
    uint32_t kSizeN;
    uint32_t kSizeK;

    if constexpr (!useProfiling)
    {
        kSizeM = 4096;
        kSizeN = 4096;
        kSizeK = 4096;
    }
    else
    {
        kSizeM = 4096 / 2;
        kSizeN = 4096 / 2;
        kSizeK = 4096 / 2;
    }

    const ktt::DimensionVector ndRangeDimensions(kSizeM, kSizeN);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(8, 8);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

    std::vector<float> mat_a(kSizeM * kSizeK);
    std::vector<float> mat_b(kSizeN * kSizeK);
    std::vector<float> mat_c(kSizeM * kSizeN);

    for (uint32_t i = 0; i < kSizeM * kSizeK; ++i)
    {
        mat_a[i] = distribution(engine);
    }

    for (uint32_t i = 0; i < kSizeN * kSizeK; ++i)
    {
        mat_b[i] = distribution(engine);
    }

    for (uint32_t i = 0; i < kSizeM * kSizeN; ++i)
    {
        mat_c[i] = 0.0f;
    }

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    if constexpr (useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("gemm_fast", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("gemm_reference", referenceKernelFile,
        ndRangeDimensions, referenceWorkGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Gemm", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("GemmReference", referenceDefinition);

    if constexpr (useReducedSet)
    {
        tuner.AddParameter(kernel, "MWG", std::vector<uint64_t>{16, 32, 64});
        tuner.AddParameter(kernel, "NWG", std::vector<uint64_t>{16, 32, 64});
        tuner.AddParameter(kernel, "KWG", std::vector<uint64_t>{32});
        tuner.AddParameter(kernel, "MDIMC", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "NDIMC", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "MDIMA", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "NDIMB", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "KWI", std::vector<uint64_t>{2});
        tuner.AddParameter(kernel, "VWM", std::vector<uint64_t>{1, 2, 4});
        tuner.AddParameter(kernel, "VWN", std::vector<uint64_t>{1, 2, 4});
        tuner.AddParameter(kernel, "STRM", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "STRN", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "SA", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "SB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "PRECISION", std::vector<uint64_t>{32});
    }
    else
    {
        tuner.AddParameter(kernel, "MWG", std::vector<uint64_t>{16, 32, 64, 128});
        tuner.AddParameter(kernel, "NWG", std::vector<uint64_t>{16, 32, 64, 128});
        tuner.AddParameter(kernel, "KWG", std::vector<uint64_t>{16, 32});
        tuner.AddParameter(kernel, "MDIMC", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "NDIMC", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "MDIMA", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "NDIMB", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "KWI", std::vector<uint64_t>{2, 8});

        if constexpr (computeApi == ktt::ComputeApi::OpenCL)
        {
            tuner.AddParameter(kernel, "VWM", std::vector<uint64_t>{1, 2, 4, 8});
            tuner.AddParameter(kernel, "VWN", std::vector<uint64_t>{1, 2, 4, 8});
        }
        else
        {
            tuner.AddParameter(kernel, "VWM", std::vector<uint64_t>{1, 2, 4});
            tuner.AddParameter(kernel, "VWN", std::vector<uint64_t>{1, 2, 4});
        }

        tuner.AddParameter(kernel, "STRM", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "STRN", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "SA", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "SB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "PRECISION", std::vector<uint64_t>{32});
    }

    // Add kernel dimension modifiers based on added tuning parameters
    auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector) {return size * vector.at(0) / vector.at(1);};
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"MDIMC", "MWG"}, globalModifier);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"NDIMC", "NWG"}, globalModifier);

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "MDIMC", ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "NDIMC", ktt::ModifierAction::Multiply);

    // Add all arguments utilized by kernels
    const ktt::ArgumentId kSizeMId = tuner.AddArgumentScalar(kSizeM);
    const ktt::ArgumentId kSizeNId = tuner.AddArgumentScalar(kSizeN);
    const ktt::ArgumentId kSizeKId = tuner.AddArgumentScalar(kSizeK);
    const ktt::ArgumentId matAId = tuner.AddArgumentVector(mat_a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId matBId = tuner.AddArgumentVector(mat_b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId matCId = tuner.AddArgumentVector(mat_c, ktt::ArgumentAccessType::WriteOnly);

    // Add conditions
    auto multipleOfX = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], v[1]);};
    auto multipleOfXMulY = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], v[1] * v[2]);};
    auto multipleOfXMulYDivZ = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], (v[1] * v[2]) / v[3]);};

    tuner.AddConstraint(kernel, {"KWG", "KWI"}, multipleOfX);
    tuner.AddConstraint(kernel, {"MWG", "MDIMC", "VWM"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"NWG", "NDIMC", "VWN"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"MWG", "MDIMA", "VWM"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"NWG", "NDIMB", "VWN"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, multipleOfXMulYDivZ);
    tuner.AddConstraint(kernel, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, multipleOfXMulYDivZ);

    tuner.SetArguments(definition, {kSizeMId, kSizeNId, kSizeKId, matAId, matBId, matCId});
    tuner.SetArguments(referenceDefinition, {kSizeMId, kSizeNId, kSizeKId, matAId, matBId, matCId});

    if constexpr (!rapidTest)
    {
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001);
        tuner.SetReferenceKernel(matCId, referenceKernel, ktt::KernelConfiguration());
    }

    // The stock example pre-heated with tuner.Tune(kernel, TuningDuration(300))
    // and then re-tuned with tuner.Tune(kernel). Both are replaced by one
    // library-driven loop: TuneIteration() benchmarks exactly ONE
    // configuration per call, we feed the measured duration + measured
    // overhead to the CI tuner, and it answers how many more tuning steps are
    // worth performing (0 = stop). The remaining X budget is spent running
    // the GEMM with the best configuration found.
    KernelConfig cfg = {};
    cfg.struct_size = sizeof(KernelConfig);
    cfg.total_kernel_runs = totalRuns; // X
    cfg.overhead = cfgOverhead;        // feeds the HISTORICAL fit; live cost model
                                       // uses per-step measured totals (push_result)
    cfg.fit_start = 10;
    // Historical data source: resolved at init from the PRECOMPUTED table
    // (historical_params.csv, generated by src/precompute_historical.py) with
    // a compute fallback on miss. See file header for TUNER_HISTORICAL_PARAMS.
    cfg.mode = tunerMode;
    cfg.k = k;
    // Historical data source: resolved at init from the PRECOMPUTED table
    // (historical_params.csv, generated by src/precompute_historical.py) with
    // a compute fallback on miss. See file header for TUNER_HISTORICAL_PARAMS.
    cfg.hist_HW = histHW.c_str();
    cfg.file_name = "gemm-reduced_output.csv";
    cfg.hist_number_of_tests = 1000;

    TunerStatus status = TUNER_OK;
    KernelHandle* handle = initiate_kernel(&cfg, "ClTuneGemm", &status);
    if (handle == nullptr || status != TUNER_OK)
    {
        std::cerr << "initiate_kernel failed, status " << status << "\n";
        return 1;
    }

    // Searcher choice:
    //   det (default) -- DeterministicSearcher, FIXED order: a "ref" session
    //     and a tuning session sample the SAME stream (the tuner session is a
    //     prefix of the reference), so decline% = T_est/T* - 1 is paired and
    //     >= 0 by construction. NOTE: lexicographic order is NOT random --
    //     it front-loads MWG=16 configs -- so paired-stream results carry a
    //     stream-order bias.
    //   rand -- RandomSearcher, deployment-style sampling with replacement
    //     from the whole pool: the same scheme the work_* harness simulations
    //     assume. No stream pairing (T* must come from an EXHAUSTIVE ref
    //     diary, which bounds every possible stream); this is the
    //     apples-to-apples comparison against the harness numbers.
    if (useRandomSearcher)
        tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
    else
        tuner.SetSearcher(kernel, std::make_unique<ktt::DeterministicSearcher>());

    // ---- REFERENCE MODE (mode "ref") --------------------------------------
    // Pure random search for R = X iterations, NO CI tuner. Produces
    // reference_steps_<tag>.csv in the same format as tuning_steps so the
    // comparison script can compute the oracle-proxy session cost T*(X)
    // (the best achievable stop on this hardware) and the real decline% of
    // a CI-tuner session against it -- the real-hardware twin of the
    // work_*/seeds_* C_decl% / Py_decl% columns.
    if (modeStr == "ref")
    {
        std::ofstream refLog("reference_steps_" + tag + ".csv");
        refLog << "step,kernel_us,overhead_us,total_us,new_best\n";
        double bestRefUs = std::numeric_limits<double>::infinity();
        std::vector<ktt::KernelResult> refResults;
        uint64_t refFailed = 0;
        for (uint64_t step = 1; step <= totalRuns; ++step)
        {
            const auto t0 = std::chrono::steady_clock::now();
            ktt::KernelResult result = tuner.TuneIteration(kernel, {});
            const auto t1 = std::chrono::steady_clock::now();
            if (result.GetStatus() != ktt::ResultStatus::Ok)
            {
                if (++refFailed > 1000) { std::cerr << "too many failures\n"; break; }
                continue;
            }
            const double kernelUs = static_cast<double>(result.GetKernelDuration()) / 1000.0;
            const double wallUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
            refResults.push_back(result);
            refLog << step << "," << kernelUs << "," << wallUs - kernelUs << ","
                   << wallUs << "," << (kernelUs < bestRefUs ? 1 : 0) << "\n";
            if (kernelUs < bestRefUs)
                bestRefUs = kernelUs;
            std::cout << "ref step " << step << "/" << totalRuns << ": kernel "
                      << kernelUs << " us" << (kernelUs == bestRefUs ? "  <-- best" : "") << "\n";
        }
        refLog.close();
        if (!refResults.empty())
        {
            const ktt::KernelResult best = tuner.GetBestResult(refResults);
            std::cout << "Reference search done: best kernel " << bestRefUs
                      << " us (config: " << best.GetConfiguration().GetString() << ")\n";
            tuner.SaveResults(refResults, "GemmOutput_" + tag, ktt::OutputFormat::XML);
        }
        return 0;
    }

    std::vector<ktt::KernelResult> tuningResults;
    tuningResults.reserve(totalRuns);

    uint64_t runsDone = 0;      // executions consumed from X
    uint64_t tuningSteps = 0;   // of those, how many were tuning steps
    uint64_t failedIterations = 0;
    const uint64_t failureCap = 1000; // safety against a pathological config space
    bool tuning = true;
    ktt::KernelConfiguration bestConfiguration{};

    // Per-step CSV (tuning_steps.csv): step, kernel_us, overhead_us,
    // total_us, new_best, remaining budget -- the real-tuning analog of the
    // work_*/result.txt tables for offline comparison.
    std::ofstream stepLog("tuning_steps_" + tag + ".csv");
    stepLog << "step,kernel_us,overhead_us,total_us,new_best,budget\n";
    double bestKernelUs = std::numeric_limits<double>::infinity();

    while (runsDone < totalRuns)
    {
        if (tuning)
        {
            // ONE tuning iteration: benchmarks a single configuration chosen
            // by the searcher (first call also initializes the config space).
            // Wall-clock around the call is the TRUE per-step cost: kernel
            // runtime + NVRTC compilation + searcher + host overhead. The
            // CUDA backend leaves KernelResult::GetKernelOverhead() at 0, so
            // this measured total replaces the (empty) KTT field.
            const auto iterStart = std::chrono::steady_clock::now();
            ktt::KernelResult result = tuner.TuneIteration(kernel, {});
            const auto iterEnd = std::chrono::steady_clock::now();
            const double wallUs =
                std::chrono::duration<double, std::micro>(iterEnd - iterStart).count();

            if (result.GetStatus() != ktt::ResultStatus::Ok)
            {
                // Failed configs don't produce usable timing data; they also
                // don't count as GEMM executions. Guard against a space that
                // never succeeds.
                if (++failedIterations > failureCap)
                {
                    std::cerr << "Too many failed tuning iterations, aborting tuning phase\n";
                    break;
                }
                continue;
            }
            // total_time = measured full cost of trying this config (wall
            // clock); overhead = wall clock minus the kernel's own runtime.
            const double kernelTimeUs = static_cast<double>(result.GetKernelDuration()) / 1000.0;
            const double overheadUs = wallUs - kernelTimeUs;
            push_result(handle, kernelTimeUs, wallUs);
            ++runsDone;
            ++tuningSteps;
            tuningResults.push_back(result);

            // Per-step log: one row per tuning iteration, for offline
            // comparison against the validate_total_runtime harness results
            // (work_* directories) in Python. Cumulative columns mirror the
            // cost model the library optimizes.
            stepLog << tuningSteps << "," << kernelTimeUs << "," << overheadUs
                    << "," << kernelTimeUs + overheadUs << ","
                    << (kernelTimeUs < bestKernelUs ? 1 : 0) << ","
                    << find_number_of_steps_that_should_be_tuning(handle) << "\n";
            if (kernelTimeUs < bestKernelUs)
                bestKernelUs = kernelTimeUs;

            const uint64_t remainingBudget = find_number_of_steps_that_should_be_tuning(handle);
            std::cout << "step " << tuningSteps << ": kernel " << kernelTimeUs << " us, overhead "
                      << overheadUs << " us, budget " << remainingBudget << "\n";

            if (remainingBudget == 0)
            {
                tuning = false;
                bestConfiguration = tuner.GetBestConfiguration(kernel);
                std::cout << "CI tuner stopped tuning after " << tuningSteps << " steps; "
                          << "spending the remaining " << (totalRuns - runsDone)
                          << " executions with the best configuration\n";
            }
        }
        else
        {
            if (fastMode)
            {
                // Production conveys no decision information; the session
                // cost is computed arithmetically from the tuning log, so
                // end the experiment here (multi-seed testing mode).
                break;
            }
            // Production phase: run the actual GEMM with the best config.
            (void)tuner.Run(kernel, bestConfiguration, {});
            ++runsDone;
        }
    }

    if (tuning)
    {
        // X was exhausted while still tuning (library never said stop).
        std::cout << "Execution budget spent while still tuning; using best configuration found so far\n";
        if (!tuningResults.empty())
        {
            bestConfiguration = tuner.GetBestConfiguration(kernel);
        }
    }

    if (!tuningResults.empty())
    {
        const ktt::KernelResult best = tuner.GetBestResult(tuningResults);
        std::cout << "Best tuning result: kernel " << best.GetKernelDuration() / 1000.0
                  << " us (config: " << best.GetConfiguration().GetString() << ")\n";
        std::cout << "Summary: X=" << totalRuns << ", tuning steps=" << tuningSteps
                  << ", production runs=" << (runsDone - tuningSteps) << "\n";
        tuner.SaveResults(tuningResults, "GemmOutput_" + tag, ktt::OutputFormat::XML);
    }
    else
    {
        std::cerr << "No successful tuning iterations completed\n";
    }

    release_kernel(handle);
    return 0;
};
