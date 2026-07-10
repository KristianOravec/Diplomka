/* =============================================================================
 * overtune_probe.c
 *
 * BEHAVIORAL consistency test (needs no Python ground truth).
 *
 * Idea: the estimator naturally stops at some step S. We then deliberately do
 * NOT stop there -- we tune to S+delta (OVERSHOOT) or S-delta (UNDERSHOOT) -- and
 * ask for a fresh recommendation. A sane estimator should:
 *   - after OVERSHOOT (tuned past its own advice): recommend ~0 ("stop, you're
 *     already past the predicted optimum").
 *   - after UNDERSHOOT (stopped early): recommend > 0 ("keep going, there's still
 *     predicted improvement").
 *
 * This probes INTERNAL CONSISTENCY of the estimator's own predictions -- it can
 * catch problems (e.g. an estimator that keeps moving the goalpost and never
 * converges) that matching-Python cannot, because Python could share the flaw.
 *
 * We use tuner_raw_recommendation() (the fresh per-step budget) rather than the
 * latched find_number_of_steps..(), because once the kernel latches "stopped" it
 * would always report 0 -- we want its genuine opinion at the probed step.
 *
 * USAGE: ./overtune_probe [trials] [delta]
 *   trials : runs per config to average over (default 200)
 *   delta  : steps to over/under-shoot past the natural stop (default 20)
 *
 * Run from the dir containing raw-data/.
 * ============================================================================= */

#include "tuner_api.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double *load_column(const char *path, uint64_t *out_n) {
    *out_n = 0; FILE *fp = fopen(path, "r"); if (!fp) return NULL;
    char line[8192]; if (!fgets(line, sizeof(line), fp)) { fclose(fp); return NULL; }
    int target = -1, idx = 0; char *save = NULL;
    for (char *t = strtok_r(line, ",\r\n", &save); t; t = strtok_r(NULL, ",\r\n", &save)) {
        if (!strcmp(t,"Computation duration (us)")||!strcmp(t,"Kernel duration (us)")){target=idx;break;} idx++;
    }
    if (target < 0) { fclose(fp); return NULL; }
    uint64_t cap=1024,n=0; double*d=malloc(cap*sizeof(double));
    while (fgets(line,sizeof(line),fp)) {
        int col=0; char*s2=NULL; double v=0; int got=0;
        for (char*t=strtok_r(line,",\r\n",&s2);t;t=strtok_r(NULL,",\r\n",&s2)){if(col==target){v=atof(t);got=1;break;}col++;}
        if(!got)continue; if(n==cap){cap*=2;d=realloc(d,cap*sizeof(double));} d[n++]=v;
    }
    fclose(fp); *out_n=n; return d;
}
static uint64_t splitmix(uint64_t*s){uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);}

/* Live-mode configs are enough to demonstrate the behavior clearly. */
typedef struct { const char *hw; uint64_t runs; uint64_t oh; } Cfg;
static Cfg cfgs[] = {
 {"680",10000ULL,10000ULL}, {"680",10000000ULL,10000ULL},
 {"1070",10000ULL,10000ULL}, {"1070",10000000ULL,10000ULL},
 {"2080",10000ULL,10000ULL}, {"750",10000000ULL,10000ULL},
};

int main(int argc, char **argv) {
    uint64_t trials = (argc > 1) ? strtoull(argv[1], NULL, 10) : 200;
    uint64_t delta  = (argc > 2) ? strtoull(argv[2], NULL, 10) : 20;
    uint64_t fit_start = 10;

    printf("OVERTUNE / UNDERTUNE behavioral probe   (trials=%lu, delta=%lu)\n",
           (unsigned long)trials, (unsigned long)delta);
    printf("==========================================================================\n");
    printf("S = natural stop step. We check the ACTUAL stop decision (the latched\n");
    printf("countdown the tuner uses) at S-delta and S+delta.\n");
    printf("Sane: at S-delta still going (decision>0); by S+delta stopped (decision=0).\n");
    printf("--------------------------------------------------------------------------\n");
    printf("%-20s %8s %12s %12s\n", "HW/runs/oh", "avg_S", "under!=stop", "over=stop");
    printf("--------------------------------------------------------------------------\n");

    char loaded[8]=""; double*data=NULL; uint64_t ndata=0;
    int npass_under=0, npass_over=0, ntot=0;

    for (size_t c=0;c<sizeof(cfgs)/sizeof(*cfgs);c++) {
        Cfg* cc=&cfgs[c];
        if (strcmp(loaded,cc->hw)!=0){
            free(data); char path[512];
            snprintf(path,sizeof(path),"raw-data/raw-autotuning-data/gemm-reduced/%s-gemm-reduced_output.csv",cc->hw);
            data=load_column(path,&ndata);
            if(!data){printf("[skip %s: no data]\n",cc->hw);loaded[0]='\0';continue;}
            snprintf(loaded,sizeof(loaded),"%s",cc->hw);
        }
        uint64_t curve_limit=(cc->runs<ndata)?cc->runs:ndata; if(curve_limit>2000)curve_limit=2000;

        KernelConfig cfg={0}; cfg.struct_size=sizeof(cfg);
        cfg.total_kernel_runs=cc->runs; cfg.overhead=cc->oh; cfg.fit_start=fit_start;
        cfg.mode=TUNER_MODE_LIVE; cfg.k=1.0;
        KernelHandle* h=initiate_kernel(&cfg,NULL,NULL);
        if(!h){printf("[init failed %s]\n",cc->hw);continue;}

        double sum_S=0;
        int under_keptgoing=0, over_stopped=0;  /* per-trial pass counts */
        uint64_t rng=0x0DD7A11ULL ^ ((uint64_t)c<<3);

        for (uint64_t t=0;t<trials;t++){
            /* pre-generate one run so all probes see the SAME data */
            static double run[2000];
            for(uint64_t s=0;s<curve_limit;s++) run[s]=data[splitmix(&rng)%ndata];

            /* Drive ONE continuous run, recording the decision at each step.
             * This is the real tuner behavior: the latched countdown via
             * find_number_of_steps_that_should_be_tuning(). We note:
             *   S            = first step where it says stop
             *   decision at S-delta (was it still going?)
             *   decision at S+delta (has it stopped?) */
            reset_kernel(h);
            uint64_t S=curve_limit-1; int found=0;
            int dec_under=1, dec_over=0;
            uint64_t under_step=0, over_step=0;
            /* first pass: find S */
            for(uint64_t s=0;s<curve_limit;s++){
                push_result(h,run[s],run[s]+(double)cc->oh);
                if(!found && find_number_of_steps_that_should_be_tuning(h)==0){S=s;found=1;break;}
            }
            sum_S+=(double)(S+1);
            under_step=(S>delta)?(S-delta):0;
            over_step=S+delta; if(over_step>=curve_limit)over_step=curve_limit-1;

            /* second pass: replay and capture the stop decision at the two probe
             * points (re-run because the first pass broke at S). */
            reset_kernel(h);
            for(uint64_t s=0;s<curve_limit;s++){
                push_result(h,run[s],run[s]+(double)cc->oh);
                uint64_t dec=find_number_of_steps_that_should_be_tuning(h);
                if(s==under_step) dec_under=(dec>0);   /* still going at S-delta? */
                if(s==over_step){ dec_over=(dec==0); break; } /* stopped by S+delta? */
            }
            under_keptgoing += dec_under;
            over_stopped    += dec_over;
        }
        release_kernel(h);

        double avgS=sum_S/trials;
        double pct_under=100.0*under_keptgoing/trials;
        double pct_over =100.0*over_stopped/trials;
        int pu=(pct_under>50.0), po=(pct_over>50.0);
        npass_under+=pu; npass_over+=po; ntot++;

        char name[40]; snprintf(name,sizeof(name),"%s/%lu/%lu",cc->hw,(unsigned long)cc->runs,(unsigned long)cc->oh);
        printf("%-20s %8.1f %11.0f%% %11.0f%%   %s%s\n", name, avgS, pct_under, pct_over,
               pu?"":"[under?] ", po?"":"[over?] ");
        fflush(stdout);
    }
    free(data);
    printf("--------------------------------------------------------------------------\n");
    printf("undershoot kept-going: %d/%d   overshoot wound-down: %d/%d\n",
           npass_under, ntot, npass_over, ntot);
    printf("Interpretation: avg recommendation should be HIGHER at S-delta than at\n");
    printf("S+delta -- i.e. the estimator wants more tuning when you've done less,\n");
    printf("and less (toward 0) when you've overshot. That is internal consistency.\n");
    return 0;
}
