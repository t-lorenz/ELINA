#ifndef ELINA_MAXPOOL_BOOPATHY_APPROX_H
#define ELINA_MAXPOOL_BOOPATHY_APPROX_H

#ifdef __cplusplus
extern "C" {
#endif

#include "backsubstitute.h"

size_t handle_boopathy_maxpool_layer(elina_manager_t *man, elina_abstract0_t *element, size_t *input_size,
                                     size_t *output_size, size_t *predecessors, size_t num_predecessors);


#ifdef __cplusplus
}
#endif
#endif //ELINA_MAXPOOL_BOOPATHY_APPROX_H
