/*
 * Main.cpp
 */

#include "Antenna.h"
#include "Function.h"
#include "MathHelper.h"
#include "SequentialSwarm.h"
#include "SimpleAntenna.h"
#include "SimpleExpFunction.h"
#include "Swarm.h"
#include "consts.h"

#include <fstream>
#include <cstring>
#include <iostream>
#include <stdlib.h>

#ifdef _OPENMP
	#include <omp.h>
	#include "ParallelSwarm.h"
#endif

using namespace std;

double rnd() {
	return random() % 1000000 / 1000000.0;
}

void initialize_swarm(Swarm *swarm, Function *function, int robots) {
	int dims = function->dimensions();
	double *size = new double[dims];

	for (int d = 0; d < dims; d++)
		size[d] = function->get_max_position()[d]
				- function->get_min_position()[d];

	srandom(SEED);

	for (int robot = 0; robot < robots; robot++) {
		for (int d = 0; d < dims; d++)
			swarm->set_position(d, robot,
					function->get_min_position()[d] + size[d] * rnd());
	}

	delete[] size;
}

Function* initialize_function() {
	int DIMS = 4;
	SimpleExpFunction *f = new SimpleExpFunction(DIMS, 12);

	f->set_exp_params(0, 2.0, 0.25);
	f->set_exp_params(1, 2.0, 0.3);
	f->set_exp_params(2, 2.0, 0.4);
	f->set_exp_params(3, 2.0, 0.3);
	f->set_exp_params(4, 2.5, 0.25);
	f->set_exp_params(5, 2.4, 0.3);
	f->set_exp_params(6, 2.3, 0.4);
	f->set_exp_params(7, 2.2, 0.3);
	f->set_exp_params(8, 1.5, 0.25);
	f->set_exp_params(9, 1.4, 0.3);
	f->set_exp_params(10, 1.3, 0.4);
	f->set_exp_params(11, 1.2, 0.3);

	f->set_exp_location(0, 0, 1);
	f->set_exp_location(0, 1, 1);
	f->set_exp_location(0, 2, 0.5);
	f->set_exp_location(0, 3, 1.5);

	f->set_exp_location(1, 0, 4);
	f->set_exp_location(1, 1, 4);
	f->set_exp_location(1, 2, 0.5);
	f->set_exp_location(1, 3, 3.5);

	f->set_exp_location(2, 0, 1);
	f->set_exp_location(2, 1, 4);
	f->set_exp_location(2, 2, 0.5);
	f->set_exp_location(2, 3, 1.5);

	f->set_exp_location(3, 0, 4);
	f->set_exp_location(3, 1, 1);
	f->set_exp_location(3, 2, 0.5);
	f->set_exp_location(3, 3, 3.5);

	f->set_exp_location(4, 0, 1);
	f->set_exp_location(4, 1, 1);
	f->set_exp_location(4, 2, 2.5);
	f->set_exp_location(4, 3, 1.5);

	f->set_exp_location(5, 0, 4);
	f->set_exp_location(5, 1, 4);
	f->set_exp_location(5, 2, 2.5);
	f->set_exp_location(5, 3, 3.5);

	f->set_exp_location(6, 0, 1);
	f->set_exp_location(6, 1, 4);
	f->set_exp_location(6, 2, 2.5);
	f->set_exp_location(6, 3, 1.5);

	f->set_exp_location(7, 0, 4);
	f->set_exp_location(7, 1, 1);
	f->set_exp_location(7, 2, 2.5);
	f->set_exp_location(7, 3, 3.5);

	f->set_exp_location(8, 0, 1);
	f->set_exp_location(8, 1, 1);
	f->set_exp_location(8, 2, 2.5);
	f->set_exp_location(8, 3, 1.5);

	f->set_exp_location(9, 0, 4);
	f->set_exp_location(9, 1, 4);
	f->set_exp_location(9, 2, 2.5);
	f->set_exp_location(9, 3, 3.5);

	f->set_exp_location(10, 0, 1);
	f->set_exp_location(10, 1, 4);
	f->set_exp_location(10, 2, 2.5);
	f->set_exp_location(10, 3, 1.5);

	f->set_exp_location(11, 0, 4);
	f->set_exp_location(11, 1, 1);
	f->set_exp_location(11, 2, 2.5);
	f->set_exp_location(11, 3, 3.5);

	return f;
}

void show_report(int step, Swarm *swarm, int robots, int dims) {
	for (int r = 0; r < robots; r++) {
		cout << step << " " << r << " ";
		for (int d = 0; d < dims; d++) {
			cout << swarm->get_position(r, d) << " ";
		}
		cout << endl;
	}
}

void clean(int *ptr, int size) {
	for (int i = 0; i < size; i++)
		ptr[i] = 0;
}

void create_and_show_histogram(Swarm *swarm, int robots, int histogram_size,
		double length) {
	int *h0 = new int[histogram_size];
	int *h1 = new int[histogram_size];
	clean(h0, histogram_size);
	clean(h1, histogram_size);
	double d2idx = (histogram_size - 1) / length;

	swarm->distributionOfRobots(h0, h1, histogram_size, d2idx);

	int cumulative_0 = 0;
	int cumulative_1 = 0;
	int of_all_0;
	int of_all_1;
	for (int idx = 0; idx < histogram_size; idx++) {
		cumulative_0 += h0[idx];
		cumulative_1 += h1[idx];
		of_all_0 = 100.0 * cumulative_0 / robots;
		of_all_1 = 100.0 * cumulative_1 / robots;
		cout << idx << " " << h0[idx] << " " << of_all_0 << "% " << h1[idx] << " "
				<< of_all_1 << "%" << endl;
	}

	delete[] h0;
	delete[] h1;
}

int main(int argc, char **argv) {
	double start;

	Function *function = initialize_function();
	double max_distance = MathHelper::distance(function->get_max_position(),
			function->get_min_position(), function->dimensions());

	Antenna *antenna = new SimpleAntenna(TARGET_NEIGHBOURS,
			max_distance / ANTENNA_MIN_RANGE_DIV,
			max_distance / ANTENNA_MAX_RANGE_DIV, ANTENNA_RANGE_MODIFIER);

	Swarm *swarm;

#ifdef _OPENMP
	swarm = new ParallelSwarm(ROBOTS, antenna, function);
#else
	swarm = new SequentialSwarm(ROBOTS, antenna, function);
#endif

	initialize_swarm(swarm, function, ROBOTS);

	int step = 0;
	do {
		swarm->run(REPORT_PERIOD);
		step += REPORT_PERIOD;
		show_report(step, swarm, ROBOTS, function->dimensions());
		create_and_show_histogram(swarm, ROBOTS, 33, 0.007);
	} while (step < STEPS);

	return 0;
}
