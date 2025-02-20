#include "ParallelSwarm.h"

#include "MathHelper.h"

#include "Swarm.h"

#include "consts.h"

#include <math.h>

#include <iostream>

#include <omp.h>



using namespace std;



ParallelSwarm::ParallelSwarm(int robots, Antenna *antenna,

                                 Function *function)

    : Swarm(robots, antenna, function) {

  step = 0;

  allocate_memory();

  initialize_antennas();

}



void ParallelSwarm::run(int steps) { 

  for (int step = 0; step < steps; step++)

    single_step();

}



void ParallelSwarm::single_step() {

  step++;

  evaluate_function();

  find_neighbours_and_remember_best();

  move();

  fit_antenna_range();

}



void ParallelSwarm::evaluate_function() {

  // standard pragma

  #pragma omp parallel for

  for (int robot = 0; robot < robots; robot++) {

    function_value[robot] = function->value(position[robot]);

  }

}



void ParallelSwarm::find_neighbours_and_remember_best() {

  // dynamic pragma + changes in order to maintain correctness

  

  #pragma omp parallel for schedule(dynamic, 1)

  for (int robot = 0; robot < robots; robot++) {

    int local_best_id = robot; // local best id

    int local_nearest_neighbours = 0;

    double local_best_function_value = function_value[robot];

    double local_antenna_range_sq = antenna_range_sq[robot];

    double *local_position = position[robot];



    for (int other_robot = 0; other_robot < robots; other_robot++) {

      if (robot == other_robot) continue;



      // check if other_robot is within antenna range

      if (MathHelper::distanceSQ(local_position, position[other_robot], dimensions) < local_antenna_range_sq) {

        local_nearest_neighbours++;

        

        // update best function value and best id if necessary

        if (local_best_function_value < function_value[other_robot]) {

          local_best_function_value = function_value[other_robot];

          local_best_id = other_robot;

        }

      }

    }



    // write results back to shared arrays

    nearest_neighbours[robot] = local_nearest_neighbours;

    neighbour_id[robot] = local_best_id;

  }

}







void ParallelSwarm::compare_with_other_robot(int robot, int other_robot) {

  if (MathHelper::distanceSQ(my_position, position[other_robot], dimensions) <

      my_antenna_range_sq) {

    // inny robot jest w zasięgu anteny

    nearest_neighbours[robot]++;



    if (best_function_value < function_value[other_robot]) {

      // w dodatku ma lepszą wartość funkcji

      best_function_value = function_value[other_robot];

      best_id = other_robot;

    }

  }

}



void ParallelSwarm::move() {

  #pragma omp parallel for
  for (int robot = 0; robot < robots; robot++) {

    MathHelper::move(position[robot], position[neighbour_id[robot]],

                     new_position[robot], dimensions, STEP_SIZE / sqrt(step));

  }

  #pragma omp parallel for collapse(2)
  for (int robot = 0; robot < robots; robot++)

    for (int d = 0; d < dimensions; d++)

      position[robot][d] = new_position[robot][d];

}



void ParallelSwarm::distributionOfRobots( int *histogramD0, int *histogramD1, int size, double d2idx ) {

	this->histogramD0 = histogramD0;

	this->histogramD1 = histogramD1;

	histogram_size = size;

	this->d2idx = d2idx;

  // dynamic pragma

  #pragma omp parallel for schedule(dynamic, 1)

	for ( int robotA = 0; robotA < robots; robotA++ ) {

		distributionForRobot( robotA );

	}

}



void ParallelSwarm::distributionForRobot( int robotA ) {

	double dxMin = 1000000000000000000000.0;

	double dyMin = 1000000000000000000000.0;

	double dx;

	double dy;

	for ( int robotB = 0; robotB < robots; robotB++ ) {

		if ( robotA == robotB ) continue;

		dx = fabs( position[robotA][0] - position[robotB][0]);

		dy = fabs( position[robotA][1] - position[robotB][1]);



		if ( dx < dxMin ) dxMin = dx;

		if ( dy < dyMin ) dyMin = dy;

	}


  
	int idxD0 = dxMin * d2idx;

	int idxD1 = dyMin * d2idx;

  #pragma omp critical
  {

	if ( idxD0 < histogram_size )

		histogramD0[ idxD0 ]++;

	if ( idxD1 < histogram_size )

		histogramD1[ idxD1 ]++;

  }

}





void ParallelSwarm::allocate_memory() {

  position = new double *[robots];

  new_position = new double *[robots];

  for (int i = 0; i < robots; i++) {

    position[i] = new double[dimensions];

    new_position[i] = new double[dimensions];

  }



  neighbour_id = new int[robots];

  nearest_neighbours = new int[robots];

  function_value = new double[robots];

  antenna_range_sq = new double[robots];

}



void ParallelSwarm::fit_antenna_range() {

  

  # pragma omp parallel for
  for (int robot = 0; robot < robots; robot++) {

	  
    double range;
    range = antenna->range(sqrt(antenna_range_sq[robot]),

                           nearest_neighbours[robot]);

    antenna_range_sq[robot] = range * range;

  }

}



void ParallelSwarm::initialize_antennas() {

  double vSQ = antenna->initial_range();

  vSQ *= vSQ;

  // standard pragma

  #pragma omp parallel for 

  for (int r = 0; r < robots; r++) {

    antenna_range_sq[r] = vSQ;

  }

}



void ParallelSwarm::set_position(int dimension, int robot, double position) {

  this->position[robot][dimension] = position;

}



double ParallelSwarm::get_position(int robot, int dimension) {

  return position[robot][dimension];

}

