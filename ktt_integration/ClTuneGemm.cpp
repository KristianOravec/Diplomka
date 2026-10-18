// =============================================================================
// ClTuneGemm.cpp -- KTT GEMM example driven by the tuning-budget library.
//
// Instead of tuner.Tune(), one configuration is tuned per TuneIteration() call
// and pushed to the library, which decides when to stop. The rest of the X
// executions run the best configuration found.
//
// Usage:
//   ./ClTuneGemm [platform] [device] [kernelFile] [refKernelFile] <X>
//                [mode: live|hybrid|hist|ref] [histHW] [fast] [overhead_us]
//                [fit_start]
//   e.g. ./ClTuneGemm 0 0 ClTuneGemm.cu ClTuneGemmReference.cu 10000 hybrid 5050 fast 68220
//
//   ref   : random search for X steps without the library; histHW = GPU name,
//           writes <gpu>-gemm-reduced_output.csv and <gpu>-reference_steps.csv
//   fast  : stop right after tuning stops (production cost is computed offline)
//
// Limitations: output validation is off (rapidTest); failed configs are not
// pushed (their time is reported at the end).
// =============================================================================

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#include "tuner_api.h"
#include "evaluator.h"
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

// true = no output validation (much faster; a wrong-but-fast config is not rejected)
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

// One tuning step, timed: KTT picks the next config, compiles it and runs it.
// wallUs = the whole step (compile + run + KTT bookkeeping).
struct TimedStep
{
    ktt::KernelResult result;
    double kernelUs;
    double wallUs;
};

TimedStep RunTimedStep(ktt::Tuner& tuner, ktt::KernelId kernel)
{
    const auto start = std::chrono::steady_clock::now();
    ktt::KernelResult result = tuner.TuneIteration(kernel, {});
    const auto end = std::chrono::steady_clock::now();

    const double kernelUs = static_cast<double>(result.GetKernelDuration()) / 1000.0;
    const double wallUs = std::chrono::duration<double, std::micro>(end - start).count();
    return {result, kernelUs, wallUs};
}

// ---- ref mode helpers --------------------------------------------------------

// Value of one tuning parameter in a configuration, e.g. "MWG" -> "64".
std::string GetParameterValue(const ktt::KernelConfiguration& config, const std::string& name)
{
    for (const auto& pair : config.GetPairs())
    {
        if (pair.GetName() == name)
            return pair.GetValueString();
    }
    return "";
}

// Write all results as a pool CSV in Jarda's <GPU>-gemm-reduced_output.csv layout.
void WritePoolCsv(const std::string& path, const std::vector<ktt::KernelResult>& results)
{
    static const std::vector<std::string> parameterNames = {
        "MWG", "NWG", "KWG", "MDIMC", "NDIMC", "MDIMA", "NDIMB", "KWI",
        "VWM", "VWN", "STRM", "STRN", "SA", "SB", "PRECISION"};

    std::ofstream pool(path);

    // header
    pool << "Kernel name,Computation duration (us),Global size,Local size";
    for (const auto& name : parameterNames)
        pool << "," << name;
    pool << ",Maximum work-group size,Local memory size,Private memory size,"
            "Constant memory size,Registers count\n";

    // one row per tested configuration
    for (const auto& result : results)
    {
        const ktt::ComputationResult& run = result.GetResults().front();

        pool << run.GetKernelFunction() << ","
             << run.GetDuration() / 1000.0 << ","              // ns -> us
             << run.GetGlobalSize().GetTotalSize() << ","
             << run.GetLocalSize().GetTotalSize();

        for (const auto& name : parameterNames)
            pool << "," << GetParameterValue(result.GetConfiguration(), name);

        if (run.HasCompilationData())
        {
            const auto& compiled = run.GetCompilationData();
            pool << "," << compiled.m_MaxWorkGroupSize
                 << "," << compiled.m_LocalMemorySize
                 << "," << compiled.m_PrivateMemorySize
                 << "," << compiled.m_ConstantMemorySize
                 << "," << compiled.m_RegistersCount;
        }
        else
        {
            pool << ",,,,,";
        }
        pool << "\n";
    }
}

// Random search for up to totalRuns steps without the library.
// Logs every step to logPath and returns all successful results.
std::vector<ktt::KernelResult> RunReferenceSearch(ktt::Tuner& tuner, ktt::KernelId kernel,
                                                  uint64_t totalRuns, const std::string& logPath)
{
    std::ofstream log(logPath);
    log << "step,kernel_us,overhead_us,total_us,new_best\n";

    std::vector<ktt::KernelResult> results;
    double bestKernelUs = std::numeric_limits<double>::infinity();
    uint64_t failed = 0;

    for (uint64_t step = 1; step <= totalRuns; ++step)
    {
        const TimedStep timed = RunTimedStep(tuner, kernel);

        if (timed.result.GetStatus() != ktt::ResultStatus::Ok)
        {
            if (++failed > 1000)
            {
                std::cerr << "too many failures\n";
                break;
            }
            continue;                                   // skip failed configs
        }

        const double kernelUs = timed.kernelUs;
        const double wallUs = timed.wallUs;
        const double overheadUs = wallUs - kernelUs;
        const bool isNewBest = kernelUs < bestKernelUs;

        results.push_back(timed.result);
        log << step << "," << kernelUs << "," << overheadUs << ","
            << wallUs << "," << (isNewBest ? 1 : 0) << "\n";

        if (isNewBest)
            bestKernelUs = kernelUs;

        std::cout << "ref step " << step << "/" << totalRuns << ": kernel "
                  << kernelUs << " us" << (isNewBest ? "  <-- best" : "") << "\n";
    }

    if (!results.empty())
    {
        std::cout << "Reference search done: best kernel " << bestKernelUs << " us (config: "
                  << tuner.GetBestResult(results).GetConfiguration().GetString() << ")\n";
    }
    return results;
}

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

    // ---- tuner arguments ----------------------------------------------------
    // X = total GEMM executions (tuning steps + production runs). Required.
    if (argc < 6)
    {
        std::cerr << "Missing required argument: <X: number of CITuneGemm executions>\n";
        std::cerr << "Usage: " << argv[0] << " [platform] [device] [kernelFile] [referenceKernelFile]"
                  << " <X> [mode: live|hybrid|hist|ref] [histHW]"
                  << " [fast] [overhead_us] [fit_start, default 10]\n";
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
    // fast: stop after the stop decision; cost = tuning sum + (X - S) * best
    const bool fastMode = (argc >= 9 && std::string(argv[8]) == "fast");
    // overhead (us): used only for the historical values (O_hist, a, b);
    // live decisions use the measured per-step cost
    const uint64_t cfgOverhead = (argc >= 10) ? std::stoull(std::string(argv[9])) : 0;
    const uint64_t fitStart = (argc >= 11) ? std::stoull(std::string(argv[10])) : 10;
    // matrix size is part of the output name (results of different sizes don't mix)
    constexpr uint32_t kMatSize = useProfiling ? 4096 / 2 : 4096;
    // tuning_steps_<X>_<mode>_hist<hw>_rand_n<size>_fs<fit_start>[_fast][_oh<us>].csv
    std::string tag = std::to_string(totalRuns) + "_" + modeStr + "_hist" + histHW
                    + "_rand"
                    + "_n" + std::to_string(kMatSize)
                    + "_fs" + std::to_string(fitStart);
    if (fastMode) tag += "_fast";
    if (cfgOverhead > 0) tag += "_oh" + std::to_string(cfgOverhead);
    // ref mode: the 7th argument names the GPU being measured (e.g. 5050);
    // outputs are <gpu>-gemm-reduced_output.csv (Jarda's layout) and
    // <gpu>-reference_steps.csv
    const std::string refGpu = histHW;

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
    if (modeStr == "ref" && argc < 8)
    {
        std::cerr << "ref mode needs the GPU name, e.g. ... 5788 ref 5050\n";
        return 1;
    }

    // overhead 0 makes the historical values wrong (tuning treated as free)
    if ((tunerMode == TUNER_MODE_HYBRID || tunerMode == TUNER_MODE_HISTORICAL)
        && cfgOverhead == 0)
    {
        std::cerr << "WARNING: mode '" << modeStr << "' with overhead_us = 0 "
                  << "(argv[9] not given).\n"
                  << "         The historical fit will treat tuning as free and "
                  << "O_hist will be too deep.\n"
                  << "         Pass the measured per-step overhead, e.g. the "
                  << "median overhead_us of a reference sweep.\n";
    }

    std::cout << "CONFIG: X=" << totalRuns << " mode=" << modeStr
              << " hist=" << histHW << " searcher=rand"
              << " n=" << kMatSize << " fit_start=" << fitStart
              << " overhead_us=" << cfgOverhead
              << (fastMode ? " fast" : "") << "\n";

    uint32_t kSizeM;
    uint32_t kSizeN;
    uint32_t kSizeK;

    kSizeM = kMatSize;
    kSizeN = kMatSize;
    kSizeK = kMatSize;

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

    // ---- tuning-budget library ----------------------------------------------
    KernelConfig cfg = {};
    cfg.total_kernel_runs = totalRuns; // X
    cfg.overhead = cfgOverhead;        // for the historical values only
    cfg.fit_start = fitStart;
    cfg.mode = tunerMode;
    cfg.k = k;
    cfg.hist_HW = histHW.c_str();      // a, b / O_hist from historical_params.csv or computed
    cfg.file_name = "gemm-reduced_output.csv";
    cfg.hist_number_of_tests = 1000;

    TunerStatus status = TUNER_OK;
    KernelHandle* handle = initiate_kernel(&cfg, "ClTuneGemm", &status);
    if (handle == nullptr || status != TUNER_OK)
    {
        std::cerr << "initiate_kernel failed, status " << status << "\n";
        return 1;
    }

    // random order, no repeats
    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());

    // ---- ref mode: random search for X steps, no library; writes the pool CSV
    if (modeStr == "ref")
    {
        const std::vector<ktt::KernelResult> refResults =
            RunReferenceSearch(tuner, kernel, totalRuns, refGpu + "-reference_steps.csv");

        if (!refResults.empty())
        {
            tuner.SaveResults(refResults, refGpu + "-GemmOutput", ktt::OutputFormat::XML);

            const std::string poolPath = refGpu + "-gemm-reduced_output.csv";
            WritePoolCsv(poolPath, refResults);
            std::cout << "Pool written: " << poolPath << "\n";
        }
        release_kernel(handle);
        return 0;
    }

    std::vector<ktt::KernelResult> tuningResults;
    tuningResults.reserve(totalRuns);

    uint64_t runsDone = 0;      // executions used from X
    uint64_t tuningSteps = 0;   // of those, tuning steps
    uint64_t failedIterations = 0;
    double failedWallUs = 0.0;  // time of failed configs (not seen by the library)
    const uint64_t failureCap = 1000;
    bool tuning = true;
    ktt::KernelConfiguration bestConfiguration{};

    std::ofstream stepLog("tuning_steps_" + tag + ".csv");
    stepLog << "step,kernel_us,overhead_us,total_us,new_best,budget,raw_est\n";
    double bestKernelUs = std::numeric_limits<double>::infinity();

    // step 1 is held back and pushed at step 2 (see below)
    bool havePending = false;
    double pendingKernelUs = 0.0;
    double pendingWallUs = 0.0;

    // ask the library once per push, log one CSV row, return the budget
    // (raw_est = fresh estimate that ignores the stop latch)
    auto logRow = [&](uint64_t step, double kUs, double ohUs) -> uint64_t
    {
        const uint64_t budget = find_number_of_steps_that_should_be_tuning(handle);
        const uint64_t rawEst = tuner_raw_recommendation(handle);
        stepLog << step << "," << kUs << "," << ohUs << "," << kUs + ohUs << ","
                << (kUs < bestKernelUs ? 1 : 0) << "," << budget << "," << rawEst << "\n";
        if (kUs < bestKernelUs)
            bestKernelUs = kUs;
        std::cout << "step " << step << ": kernel " << kUs << " us, overhead "
                  << ohUs << " us, budget " << budget << "\n";
        return budget;
    };

    while (runsDone < totalRuns)
    {
        if (tuning)
        {
            const TimedStep timed = RunTimedStep(tuner, kernel);
            const double wallUs = timed.wallUs;

            if (timed.result.GetStatus() != ktt::ResultStatus::Ok)
            {
                // failed config: no timing, not a GEMM execution
                failedWallUs += wallUs;
                if (++failedIterations > failureCap)
                {
                    std::cerr << "Too many failed tuning iterations, aborting tuning phase\n";
                    break;
                }
                continue;
            }
            const double kernelTimeUs = timed.kernelUs;
            const double overheadUs = wallUs - kernelTimeUs;
            ++runsDone;
            ++tuningSteps;
            tuningResults.push_back(timed.result);

            // step 1 includes one-off setup (~3x a normal step): push it later
            // with step 2's overhead; the CSV keeps the real value
            if (tuningSteps == 1)
            {
                havePending = true;
                pendingKernelUs = kernelTimeUs;
                pendingWallUs = wallUs;
                continue;
            }
            if (havePending)
            {
                push_result(handle, pendingKernelUs, pendingKernelUs + overheadUs);
                (void)logRow(1, pendingKernelUs, pendingWallUs - pendingKernelUs);
                havePending = false;
            }
            push_result(handle, kernelTimeUs, wallUs);
            const uint64_t remainingBudget = logRow(tuningSteps, kernelTimeUs, overheadUs);

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
                break;   // production cost is computed from the CSV
            }
            // production run with the best configuration
            (void)tuner.Run(kernel, bestConfiguration, {});
            ++runsDone;
        }
    }

    // no step 2 happened: push step 1 with its real cost
    if (havePending)
    {
        push_result(handle, pendingKernelUs, pendingWallUs);
        (void)logRow(1, pendingKernelUs, pendingWallUs - pendingKernelUs);
        havePending = false;
    }

    if (tuning)
    {
        // X ran out before the library said stop
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
        if (failedIterations > 0)
            std::cout << "Failed configurations: " << failedIterations
                      << " (" << failedWallUs / 1e6 << " s wall time, NOT seen by "
                      << "the cost model)\n";
        tuner.SaveResults(tuningResults, "GemmOutput_" + tag, ktt::OutputFormat::XML);
    }
    else
    {
        std::cerr << "No successful tuning iterations completed\n";
    }

    release_kernel(handle);
    return 0;
};
