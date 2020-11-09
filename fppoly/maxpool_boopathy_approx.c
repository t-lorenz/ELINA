#include <stdlib.h>
#include "maxpool_boopathy_approx.h"

size_t handle_boopathy_maxpool_layer(elina_manager_t *man, elina_abstract0_t *element, size_t *input_size,
                                     size_t *output_size, size_t *predecessors, size_t num_predecessors) {

    assert(num_predecessors == 1);

    // order of dimension is H x W x C. We only support global pooling across H for each C with W=1.
    assert(input_size[1] == 1);
    assert(output_size[0] == 1);
    assert(output_size[1] == 1);

    size_t num_input_neurons = input_size[0] * input_size[1] * input_size[2];
    size_t num_out_neurons = output_size[0] * output_size[1] * output_size[2];
    // number of values over which each max operation is computed
    size_t max_pool_size = input_size[0];

    fppoly_t *fp = fppoly_of_abstract0(element);
    size_t current_layer_index = fp->numlayers;
    fppoly_add_new_layer(fp, num_out_neurons, predecessors, num_predecessors, false);

    // lower bounds of inputs
    double *inf = (double *) calloc(max_pool_size, sizeof(double));
    // upper bounds of inputs
    double *sup = (double *) calloc(max_pool_size, sizeof(double));
    // map from flattened pooling index to original input variable index
    size_t *pool_map = (size_t *) calloc(max_pool_size, sizeof(size_t));

    neuron_t **out_neurons = fp->layers[current_layer_index]->neurons;
    int previous_layer_index = predecessors[0] - 1;
    neuron_t **in_neurons = fp->layers[previous_layer_index]->neurons;

    // loop over all output neurons. In our case this is equivalent of iterating over C dimension. All inner loops
    // iterate over inputs to one single output - in our case simply the H dimension.
    size_t out_pos;
    for (out_pos = 0; out_pos < num_out_neurons; out_pos++) {
        double max_upper_bound = -INFINITY;
        double min_upper_bound = INFINITY;
        double max_lower_bound = -INFINITY;
        double min_lower_bound = INFINITY;

        // Find max lower bound
        size_t i;
        for (i = 0; i < max_pool_size; i++) {
            size_t input_index = i * max_pool_size + out_pos;
            double lower_bound = -in_neurons[input_index]->lb;

            if (lower_bound > max_lower_bound) {
                max_lower_bound = lower_bound;
            }
        }

        double gamma = INFINITY;
        int num_equal_bounds = 0;

        // Determine actual inputs to pooling function. All inputs with ub < max_lb are ignored
        // as they can never be the max value.
        size_t actual_pool_size = 0;
        for (i = 0; i < max_pool_size; i++) {
            size_t input_index = i * max_pool_size + out_pos;
            double lower_bound = -in_neurons[input_index]->lb;
            double upper_bound = in_neurons[input_index]->ub;

            if (upper_bound >= max_lower_bound) {
                inf[actual_pool_size] = lower_bound;
                sup[actual_pool_size] = upper_bound;
                pool_map[actual_pool_size] = input_index;

                if (upper_bound > max_upper_bound) {
                    max_upper_bound = upper_bound;
                }
                if (upper_bound < min_upper_bound) {
                    min_upper_bound = upper_bound;
                }
                if (lower_bound < min_lower_bound) {
                    min_lower_bound = lower_bound;
                }

                if (lower_bound == upper_bound) {
                    gamma = lower_bound;
                    num_equal_bounds += 1;
                }

                actual_pool_size++;
            }
        }

        // compute weights for each input variable and the corresponding index
        double coefficients[actual_pool_size];
        size_t dimensions[actual_pool_size];

        if (num_equal_bounds > 0) {
            // the case where no lb == ub - i.e. the one defined in the actual paper
            double gamma_numerator = 0.0;
            double gamma_denominator = 0.0;

            for (i = 0; i < actual_pool_size; i++) {
                gamma_numerator += sup[i] / (sup[i] - inf[i]);
                gamma_denominator += 1.0 / (sup[i] - inf[i]);
            }

            gamma = (gamma_numerator - 1.0) / gamma_denominator;
            if (gamma < max_lower_bound) {
                gamma = max_lower_bound;
            } else if (gamma > min_upper_bound) {
                gamma = min_upper_bound;
            }

            for (i = 0; i < actual_pool_size; i++) {
                coefficients[i] = (sup[i] - gamma) / (sup[i] - inf[i]);
                dimensions[i] = pool_map[i];
            }
        } else {
            // the case where at least one lb == ub - not described in paper but taken from published code
            double partial_coefficient_sum = 0.0;
            for (i = 0; i < actual_pool_size; i++) {
                if (sup[i] != inf[i]) {
                    coefficients[i] = (sup[i] - gamma) / (sup[i] - inf[i]);
                    dimensions[i] = pool_map[i];
                    partial_coefficient_sum += coefficients[i];
                }
            }
            double gap = (1.0 - partial_coefficient_sum) / num_equal_bounds;
            if (gap < 0.0) {
                gap = 0.0;
            } else if (gap > 1.0) {
                gap = 1.0;
            }

            for (i = 0; i < actual_pool_size; i++) {
                if (sup[i] == inf[i]) {
                    coefficients[i] = gap;
                    dimensions[i] = pool_map[i];
                }
            }
        }

        //compute growth rate needed for lower bounds
        double growth_rate = 0.0;
        for (i = 0; i < actual_pool_size; i++) {
            growth_rate += coefficients[i];
        }

        //compute upper and lower offsets
        double upper_offset = gamma;
        for (i = 0; i < actual_pool_size; i++) {
            upper_offset -= coefficients[i] * inf[i];
        }
        double lower_offset;
        if (growth_rate <= 1.0) {
            lower_offset = min_lower_bound * (1.0 - growth_rate);
        } else {
            lower_offset = max_upper_bound * (1.0 - growth_rate);
        }

        out_neurons[out_pos]->uexpr = create_sparse_expr(coefficients, upper_offset, dimensions, actual_pool_size);
        out_neurons[out_pos]->lexpr = create_sparse_expr(coefficients, lower_offset, dimensions, actual_pool_size);
        out_neurons[out_pos]->ub = max_upper_bound;
        out_neurons[out_pos]->lb = -max_lower_bound;
    }

    free(inf);
    free(sup);
    free(pool_map);
    return num_out_neurons;
}
