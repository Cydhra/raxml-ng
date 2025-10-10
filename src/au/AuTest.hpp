#ifndef RAXML_AUTEST_HPP_
#define RAXML_AUTEST_HPP_

#include <memory>
#include "../types.hpp"
#include "../PartitionedMSA.hpp"

class AuTest {
public:
    AuTest(const std::shared_ptr<PartitionedMSA> &msa,
           const std::vector<std::vector<doubleVector> > &persite_loglh,
           const doubleVector &scales,
           const uintVector &num_replicates,
           long seed);

    virtual ~AuTest() {
        if (!test_statistics) return;

        // free individual matrices
        for (unsigned int i = 0; i < scales.size(); i++) {
            if (!test_statistics[i]) continue;
            free(test_statistics[i]);
        }
        free(test_statistics);
    };

    /**
     * Estimate the signed distance and curvature parameters, which can be used to estimate the AU p-value, among other
     * p-values.
     */
    void estimate_parameters();

    /**
     * Calculate the p-values from the previously estimated parameters.
     */
    void calculate_p_values();

private:
    const std::shared_ptr<PartitionedMSA> msa;
    const std::vector<std::vector<doubleVector> > &persite_loglh;
    double **test_statistics;
    const doubleVector &scales;
    const uintVector &num_replicates;
    const unsigned int num_trees;
    const int seed;
};

#endif //RAXML_AUTEST_HPP_
