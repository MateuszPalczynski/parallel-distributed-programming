/*
SEQUENTIAL_SWARM.h
*/

#ifndef SEQUENTIAL_SWARM_H_
#define SEQUENTIAL_SWARM_H_

#include "Antenna.h"
#include "Function.h"
#include "Swarm.h"

class SequentialSwarm: public Swarm {
    private: 
      int best_id;
      int step;
      double best_function_value;
      double my_antenna_range_sq;
      double *my_position;
      int *neighbour_id; // identyfikator najbliższego sąsiada
      int *nearest_neighbours; // liczba sąsiadów w zasięgu
      double *antenna_range_sq; // kwadrat zasięgu anteny
      double *function_value;  // wartość funkcji
      int *histogramD0;
      int *histogramD1;
      double d2idx;
      int histogram_size;

      void single_step();
      void evaluate_function();
      void find_neighbours_and_remember_best();
      void compare_with_other_robot( int robot, int other_robot );
      void move();
      void fit_antenna_range();
      void allocate_memory();
      void initialize_antennas();
      void distributionForRobot( int robot );
    public:
      SequentialSwarm( int robots, Antenna *antenna, Function *function  );
      void run( int steps );
      void set_position( int dimension, int robot, double position );
      double get_position( int robot, int dimension );
      void distributionOfRobots( int *histogramP0, int *histogramP1, int size, double d2idx );
};

#endif
