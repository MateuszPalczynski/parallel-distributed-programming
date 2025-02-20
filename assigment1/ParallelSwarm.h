/*
PARALLEL_SWARM.h
*/

#ifndef PARALLEL_SWARM_H_
#define PARALLEL_SWARM_H_

#include "Antenna.h"

#include "Function.h"

#include "Swarm.h"

class ParallelSwarm: public Swarm {
  private: int best_id;
  int step;
  double best_function_value;
  double my_antenna_range_sq;
  double * my_position;
  int * neighbour_id;
  int * nearest_neighbours;
  double * antenna_range_sq;
  double * function_value;

  void single_step();
  void evaluate_function(int my_first_robot, int my_last_robot);
  void find_neighbours_and_remember_best(int my_first_robot, int my_last_robot);
  void compare_with_other_robot(int robot, int other_robot);
  void move(int my_first_robot, int my_last_robot, int robots_per_process, int size);
  void fit_antenna_range(int my_first_robot, int my_last_robot);
  void allocate_memory();
  void initialize_antennas();
  void calculate_chunks(int rank, int size, int robots, int & robots_per_process, int & my_first_robot, int & my_last_robot);
  public: ParallelSwarm(int robots, Antenna * antenna, Function * function);
  void run(int steps);
  void before_first_run() {};
  void before_get_position() {};
  void set_position(int dimension, int robot, double position);
  double get_position(int robot, int dimension);
};

#endif