import numpy as np


def fill_missing_values_synchrony(synchrony_totalvideo):
    synchrony_totalvideo_optimized = synchrony_totalvideo.copy()
    for column in synchrony_totalvideo_optimized.T[1:]:
        for i in range(len(column)):
            if column[i] == -1:  # if element not available
                idx_next_num_avail = min([x for x in range(i, len(column)) if column[x] > -1],
                                         default=-1)  # find next non-na element
                idx_previous_num_avail = max([x for x in range(0, i) if column[x] > -1],
                                             default=-1)  # find previous non-na element
                if idx_next_num_avail != -1 and idx_previous_num_avail != -1:  # both indexes available
                    column[i] = (column[int(idx_next_num_avail)] + column[int(idx_previous_num_avail)]) / 2
                elif idx_previous_num_avail != -1:
                    column[i] = column[idx_previous_num_avail]
                elif idx_next_num_avail != -1:
                    column[i] = column[idx_next_num_avail]
                else:
                    column[i] = -1
    return synchrony_totalvideo_optimized
