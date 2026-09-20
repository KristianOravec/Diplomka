/* =============================================================================
 * validate_extended.c  -- compare C against the Python EXTENDED ground truth.
 *
 * Validates THREE independent quantities per config, not just the final %:
 *   - decline%   = 100 * avg_extra_runtime   (the headline metric)
 *   - est TS     = avg_estimate              (where the ESTIMATOR stops)
 *   - GT TS      = avg_crystal_ball          (where the ORACLE stops)
 *
 * Why this is stronger: a bug could make decline% coincidentally match while the
 * internals (estimate / crystal ball) are wrong. The GT TS in particular is
 * computed independently of the estimator, so matching it confirms the cost
 * model + sampling are faithful, separately from the estimator logic.
 *
 * Ground truth values are from extra_results_gemm-reduced_output.csv.
 *
 * USAGE:  ./validate_extended [trials] [mode]
 *   trials: MC trials per config (default 300; 1000 matches Python)
 *   mode  : fast (subset, default) | all | live
 *
 * Run from the directory containing
 *   raw-data/raw-autotuning-data/gemm-reduced/<HW>-gemm-reduced_output.csv
 * ============================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "evaluator.h"

typedef struct {
    const char* hw; const char* runs; const char* k; const char* oh;
    double decline;    /* 100 * Average extra runtime */
    double est_ts;     /* Average estimated TS */
    double gt_ts;      /* Average GT TS (crystal ball) */
} Ref;

/* From extra_results_gemm-reduced_output.csv. decline = 100 * "Average extra runtime". */
static Ref refs[] = {
 {"680","10000","1.0","10000",17.038522,22.475,71.932},
 {"680","10000","1.0","1000000",36.381331,16.711,5.389},
 {"680","10000","0.0","10000",6.673845,63.607,68.759},
 {"680","10000","0.0","1000000",37.520920,16.861,5.284},
 {"680","10000","0.5","10000",6.717256,60.432,68.539},
 {"680","10000","0.5","1000000",37.128403,16.693,5.302},
 {"680","10000000","1.0","10000",28.236678,59.119,960.839},
 {"680","10000000","1.0","1000000",26.158967,30.749,388.544},
 {"680","10000000","0.0","10000",4.784461,1093.833,982.924},
 {"680","10000000","0.0","1000000",6.623326,270.338,375.994},
 {"680","10000000","0.5","10000",0.913113,1314.916,924.007},
 {"680","10000000","0.5","1000000",2.991621,401.369,385.103},
 {"750","10000","1.0","10000",18.007170,22.568,95.197},
 {"750","10000","1.0","1000000",25.339461,16.839,5.992},
 {"750","10000","0.0","10000",9.188003,54.295,98.959},
 {"750","10000","0.0","1000000",25.452136,16.821,6.232},
 {"750","10000","0.5","10000",8.643246,59.660,96.308},
 {"750","10000","0.5","1000000",25.085750,16.819,6.034},
 {"750","10000000","1.0","10000",30.671944,69.628,1018.882},
 {"750","10000000","1.0","1000000",28.219934,36.612,657.563},
 {"750","10000000","0.0","10000",11.103594,379.125,1014.733},
 {"750","10000000","0.0","1000000",10.807085,178.806,636.337},
 {"750","10000000","0.5","10000",1.435452,1386.348,1024.430},
 {"750","10000000","0.5","1000000",4.199739,465.039,617.705},
 {"1070","10000","1.0","10000",17.551271,19.614,49.163},
 {"1070","10000","1.0","1000000",127.467788,16.612,2.696},
 {"1070","10000","0.0","10000",13.618893,113.188,48.503},
 {"1070","10000","0.0","1000000",127.480775,17.023,2.887},
 {"1070","10000","0.5","10000",10.176324,52.415,49.670},
 {"1070","10000","0.5","1000000",128.016067,16.634,2.752},
 {"1070","10000000","1.0","10000",40.405348,51.362,976.019},
 {"1070","10000000","1.0","1000000",33.004625,26.366,208.211},
 {"1070","10000000","0.0","10000",0.253486,1999.000,966.537},
 {"1070","10000000","0.0","1000000",7.067604,574.460,213.579},
 {"1070","10000000","0.5","10000",1.186747,1375.436,983.765},
 {"1070","10000000","0.5","1000000",5.101135,345.225,207.037},
 {"2080","10000","1.0","10000",15.815015,21.939,108.362},
 {"2080","10000","1.0","1000000",26.972473,16.724,5.669},
 {"2080","10000","0.0","10000",7.119833,66.422,107.163},
 {"2080","10000","0.0","1000000",28.495403,16.848,5.209},
 {"2080","10000","0.5","10000",8.106003,61.927,112.891},
 {"2080","10000","0.5","1000000",27.455944,16.716,5.551},
 {"2080","10000000","1.0","10000",26.602315,58.297,980.869},
 {"2080","10000000","1.0","1000000",24.751806,31.826,550.880},
 {"2080","10000000","0.0","10000",7.760319,469.194,963.528},
 {"2080","10000000","0.0","1000000",8.848318,210.281,581.450},
 {"2080","10000000","0.5","10000",1.328840,1329.125,958.155},
 {"2080","10000000","0.5","1000000",5.360631,419.451,587.707},
};

static int want(const Ref* r, const char* mode) {
    if (!strcmp(mode,"all")) return 1;
    if (!strcmp(mode,"live")) return !strcmp(r->k,"1.0");
    static const char* sub[][4] = {
        {"680","10000","1.0","10000"}, {"680","10000","0.0","10000"},
        {"680","10000","0.5","10000"}, {"1070","10000","1.0","10000"},
        {"2080","10000000","1.0","10000"}, {"750","10000","0.0","1000000"},
        {"1070","10000000","0.0","10000"},  /* the deterministic est=1999 cell */
    };
    for (size_t i=0;i<sizeof(sub)/sizeof(*sub);i++)
        if (!strcmp(r->hw,sub[i][0])&&!strcmp(r->runs,sub[i][1])&&
            !strcmp(r->k,sub[i][2])&&!strcmp(r->oh,sub[i][3])) return 1;
    return 0;
}

int main(int argc, char** argv) {
    uint64_t trials = (argc>1)?strtoull(argv[1],NULL,10):300;
    const char* mode = (argc>2)?argv[2]:"fast";

    printf("C vs Python EXTENDED ground truth   (trials=%lu, mode=%s)\n",
           (unsigned long)trials, mode);
    printf("=============================================================================\n");
    printf("%-24s | %16s | %16s | %14s\n", "HW/runs/k/oh",
           "decline%  (C/Py)", "estTS  (C/Py)", "gtTS  (C/Py)");
    printf("-----------------------------------------------------------------------------\n");

    int n = (int)(sizeof(refs)/sizeof(*refs));
    double s_decl=0, s_est=0, s_gt=0; int cnt=0;
    for (int i=0;i<n;i++) {
        Ref* r=&refs[i];
        if (!want(r,mode)) continue;
        EvaluatorParams p;
        p.HW=r->hw; p.file_name="gemm-reduced_output.csv";
        p.total_kernel_runs=strtoull(r->runs,NULL,10);
        p.hist_HW="680"; p.k=atof(r->k);
        p.overhead=strtoull(r->oh,NULL,10);
        p.fit_start=10; p.number_of_tests=trials;
        EvaluatorResult res; evaluator_run(&p,&res);

        double c_decl=100.0*res.avg_extra_runtime;
        s_decl += fabs(c_decl - r->decline);
        s_est  += fabs(res.avg_estimate - r->est_ts);
        s_gt   += fabs(res.avg_crystal_ball - r->gt_ts);
        cnt++;

        char cfg[40];
        snprintf(cfg,sizeof(cfg),"%s/%s/%s/%s",r->hw,r->runs,r->k,r->oh);
        printf("%-24s | %7.2f/%-7.2f | %7.1f/%-7.1f | %6.1f/%-6.1f\n",
               cfg, c_decl, r->decline,
               res.avg_estimate, r->est_ts,
               res.avg_crystal_ball, r->gt_ts);
        fflush(stdout);
    }
    printf("-----------------------------------------------------------------------------\n");
    if (cnt) {
        printf("mean |C-Py|:  decline %.2f pp   |   estTS %.1f steps   |   gtTS %.1f steps   (%d configs)\n",
               s_decl/cnt, s_est/cnt, s_gt/cnt, cnt);
        printf("GT TS is estimator-independent: matching it confirms the cost model + sampling.\n");
        printf("(all three small & shrinking with more trials = faithful port)\n");
    }
    return 0;
}
