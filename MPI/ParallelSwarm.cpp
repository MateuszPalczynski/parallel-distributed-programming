#include "ParallelSwarm.h"

#include "MathHelper.h"

#include "Swarm.h"

#include "consts.h"

#include <iostream>

#include <mpi.h>

using namespace std;

ParallelSwarm::ParallelSwarm(int robots, Antenna * antenna, Function * function): Swarm(robots, antenna, function) {
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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);
  MPI_Comm_size(MPI_COMM_WORLD, & size);

  // Calculating the robot scope for each process
  int robots_per_process, first_robot_idx, last_robot_idx;
  calculate_chunks(rank, size, robots, robots_per_process, first_robot_idx, last_robot_idx);

  // Synchronize positions across all processes
  for (int i = 0; i < robots; i++) {
    MPI_Bcast(position[i], dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  evaluate_function(first_robot_idx, last_robot_idx);

  // Synchronize function values across all processes
  for (int i = 0; i < size; i++) {
    int start = i * robots_per_process;
    int count = (i == size - 1) ? (robots - start) : robots_per_process;
    MPI_Bcast( & function_value[start], count, MPI_DOUBLE, i, MPI_COMM_WORLD);
  }

  find_neighbours_and_remember_best(first_robot_idx, last_robot_idx);
  move(first_robot_idx, last_robot_idx, robots_per_process, size);
  fit_antenna_range(first_robot_idx, last_robot_idx);

}

void ParallelSwarm::evaluate_function(int first_robot_idx, int last_robot_idx) {
  for (int robot = first_robot_idx; robot < last_robot_idx; robot++) {
    function_value[robot] = function -> value(position[robot]);
  }
}

void ParallelSwarm::find_neighbours_and_remember_best(int first_robot_idx, int last_robot_idx) {
  for (int robot = first_robot_idx; robot < last_robot_idx; robot++) {
    best_id = robot;
    nearest_neighbours[robot] = 0;
    best_function_value = function_value[robot];
    my_antenna_range_sq = antenna_range_sq[robot];
    my_position = position[robot];

    for (int other_robot = 0; other_robot < robots; other_robot++) {
      if (other_robot != robot) {
        compare_with_other_robot(robot, other_robot);
      }
    }
    neighbour_id[robot] = best_id;
  }
}

void ParallelSwarm::compare_with_other_robot(int robot, int other_robot) {
  if (MathHelper::distanceSQ(my_position, position[other_robot], dimensions) <
    my_antenna_range_sq) {
    // Another robot is within the antenna range
    nearest_neighbours[robot]++;

    if (best_function_value < function_value[other_robot]) {
      // In addition, it has a better function value
      best_function_value = function_value[other_robot];
      best_id = other_robot;
    }
  }
}

void ParallelSwarm::move(int first_robot_idx, int last_robot_idx, int robots_per_process, int size) {
  for (int robot = first_robot_idx; robot < last_robot_idx; robot++) {
    MathHelper::move(position[robot], position[neighbour_id[robot]],
      new_position[robot], dimensions, STEP_SIZE / sqrt(step));
  }

  // Synchronize new positions across all processes
  for (int i = 0; i < size; i++) {
    int start = i * robots_per_process;
    int count = (i == size - 1) ? (robots - start) : robots_per_process;
    for (int r = start; r < start + count; r++) {
      MPI_Bcast(new_position[r], dimensions, MPI_DOUBLE, i, MPI_COMM_WORLD);
    }
  }

  // Update positions for all robots
  for (int robot = 0; robot < robots; robot++) {
    for (int d = 0; d < dimensions; d++) {
      position[robot][d] = new_position[robot][d];
    }
  }
}

void ParallelSwarm::allocate_memory() {
  position = new double * [robots];
  new_position = new double * [robots];
  for (int i = 0; i < robots; i++) {
    position[i] = new double[dimensions];
    new_position[i] = new double[dimensions];
  }

  neighbour_id = new int[robots];
  nearest_neighbours = new int[robots];
  function_value = new double[robots];
  antenna_range_sq = new double[robots];
}

void ParallelSwarm::fit_antenna_range(int first_robot_idx, int last_robot_idx) {
  double range;
  for (int robot = first_robot_idx; robot < last_robot_idx; robot++) {
    range = antenna -> range(sqrt(antenna_range_sq[robot]),
      nearest_neighbours[robot]);
    antenna_range_sq[robot] = range * range;
  }
}

void ParallelSwarm::initialize_antennas() {
  double vSQ = antenna -> initial_range();
  vSQ *= vSQ;
  for (int r = 0; r < robots; r++) {
    antenna_range_sq[r] = vSQ;
  }
}

void ParallelSwarm::set_position(int dimension, int robot, double position) {
  this -> position[robot][dimension] = position;
}

double ParallelSwarm::get_position(int robot, int dimension) {
  return position[robot][dimension];
}

void ParallelSwarm::calculate_chunks(int rank, int size, int robots, int & robots_per_process, int & first_robot_idx, int & last_robot_idx) {
  // calculates the first, last robot and robot per process
  robots_per_process = robots / size;
  if (robots_per_process == 0) {
    robots_per_process = 1;
  }

  first_robot_idx = rank * robots_per_process;
  last_robot_idx = (rank == size - 1) ? robots : (rank + 1) * robots_per_process;
}