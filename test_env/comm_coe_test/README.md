# Communication Coefficient Test
This directory contains scripts to test communication coefficient for estimation of communication time in cost model.

## Main Idea

We use allreduce to test the communication coeffocient (comm_coe) between theoretical communication message size (MB) and real communication time (ms).

With the comm_coe, real communication time (ms) can be estimated by multiplying comm_coe to theoretical communication message size (MB):

$CommTimeEstimated(ms)=CommCoe(ms/MB)*MessageSize(MB)$

## Usage

Run ```sh test_all_comm_coe.sh``` to get the comm_coe table. This table contains the following comm_coe:

 ```comm_coe_1_8, comm_coe_1_4_0, comm_coe_1_4_1, comm_coe_1_2_0, comm_coe_1_2_1, comm_coe_2_4, comm_coe_2_2_0, comm_coe_2_2_1, comm_coe_4_2```

The first number means pp degree, the second number means communication group size, and the third number means whether the communication group is consecutive. For the situation that ```group_size == world_size```, the third number doesn't matter.

Specifically, the above comm_coe corresponds to the following communication groups:

```comm_coe_1_8:    [0 1 2 3 4 5 6 7]```

```comm_coe_1_4_0:  [0 2 4 6, 1 3 5 7]```

```comm_coe_1_4_1:  [0 1 2 3, 4 5 6 7]```

```comm_coe_1_2_0:  [0 2, 1 3, 4 6, 5 7]```

```comm_coe_1_2_1:  [0 1, 2 3, 4 5, 6 7]```

```comm_coe_2_4:    [0 1 2 3 | 4 5 6 7]```

```comm_coe_2_2_0:  [0 2, 1 3 | 4 6, 5 7]```

```comm_coe_2_2_1:  [0 2, 1 3 | 4 6, 5 7]```

```comm_coe_4_2:    [0 1 | 2 3 | 4 5 | 6 7]```