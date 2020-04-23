/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 10;  // Set the number of particles

  std::default_random_engine gen;
	
  // Create normal distributions for y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;
	
	int i;
	for (i = 0; i < num_particles; i++) {
	  double particle_x = particles[i].x;
	  double particle_y = particles[i].y;
	  double particle_theta = particles[i].theta;
	 
	  double pred_x;
	  double pred_y;
	  double pred_theta;
	
	  if (fabs(yaw_rate) < 0.0001) { // almost 0
	    pred_x = particle_x + velocity * cos(particle_theta) * delta_t;
	    pred_y = particle_y + velocity * sin(particle_theta) * delta_t;
	    pred_theta = particle_theta;
	  } else {
	    pred_x = particle_x + (velocity/yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
	    pred_y = particle_y + (velocity/yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
	    pred_theta = particle_theta + (yaw_rate * delta_t);
	  }
	  
	  normal_distribution<double> dist_x(pred_x, std_pos[0]);
	  normal_distribution<double> dist_y(pred_y, std_pos[1]);
	  normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
	  
	  particles[i].x = dist_x(gen);
	  particles[i].y = dist_y(gen);
	  particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  printf("dataAssociation() - start\n");
  if (predicted.empty() || observations.empty()) {
      return;
  }
  vector<LandmarkObs>::iterator obs_it = observations.begin();
  for (; obs_it != observations.end(); ++obs_it){
    LandmarkObs& obs = *obs_it;
    vector<LandmarkObs>::iterator lm_it = predicted.begin();
    
    // assign the first landmark as the min
    double min_distance = dist(obs.x, obs.y, (*lm_it).x, (*lm_it).y);
    LandmarkObs nearest_lm = *lm_it;
    ++lm_it; // go to the second landmark to begin the loop

    for (; lm_it != predicted.end(); ++lm_it) {
      LandmarkObs& lm = *lm_it;
      double distance = dist(obs.x, obs.y, lm.x, lm.y);
      if (distance < min_distance){
        // printf("updating distance(%f)\n", distance);
        min_distance = distance;
        nearest_lm = lm;
      }
    }
    // printf("min_distance(%f)\n", min_distance);
    obs.id = nearest_lm.id;
  }
  vector<LandmarkObs>::iterator obs_iter = observations.begin();
    for(; obs_iter != observations.end(); ++obs_iter){
        LandmarkObs& obs = *obs_iter;
        printf("obs.id(%d) ", obs.id);
    }
  printf("dataAssociation()- end\n");
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double weight_sum = 0.0;

  for (int i=0; i<num_particles; ++i){
    printf("particle-%d\n", i);
    float particle_x = particles[i].x;
    float particle_y = particles[i].y;
    float particle_theta = particles[i].theta;

    // Select only landmarks in the range according to the current particle ()
    vector<LandmarkObs> selected_landmarks;
    vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
    vector<Map::single_landmark_s>::iterator lm_it = landmark_list.begin();
    for (;lm_it != landmark_list.end(); ++lm_it){
      Map::single_landmark_s& lm = *lm_it;
      if ((fabs(particle_x - lm.x_f) <= sensor_range) 
            && (fabs(particle_y - lm.y_f) <= sensor_range)){
        selected_landmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }
    printf("num of landmarks in range is %d\n", selected_landmarks.size());

    // Transform obersvations to MAP's coordinate system
    vector<LandmarkObs> transformed_observations;
    vector<LandmarkObs>::const_iterator lmObs_it = observations.begin();
    for(;lmObs_it!=observations.end(); ++lmObs_it){
      const LandmarkObs& lmObs = *lmObs_it;
      LandmarkObs transformed_obs; // observation in map coordinate
      transformed_obs.id = lmObs.id;
      transformed_obs.x = particles[i].x + cos(particle_theta)*lmObs.x - sin(particle_theta)*lmObs.y;
      transformed_obs.y = particles[i].y + sin(particle_theta)*lmObs.x + cos(particle_theta)*lmObs.y;
      transformed_observations.push_back(transformed_obs);
    }

    // Data assosiation. Find and assign the nearest landmark to each observation
    dataAssociation(selected_landmarks, transformed_observations);

    // Update weight
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
  
    particles[i].weight = 1.0; // reset

    vector<LandmarkObs>::iterator lm_iter = selected_landmarks.begin();
    for (; lm_iter != selected_landmarks.end(); ++lm_iter){
      LandmarkObs& lm = *lm_iter;
      // printf("lm.id(%d)", lm.id);
      vector<LandmarkObs>::iterator obs_iter = transformed_observations.begin();
      for(; obs_iter != transformed_observations.end(); ++obs_iter){
        LandmarkObs& obs = *obs_iter;
        // printf("obs.id(%d)", obs.id);
        if(lm.id == obs.id){
          printf("%f, %f, %f, %f", obs.x, obs.y, lm.x, lm.y);
          double multi_prob = multiv_prob(sigma_x, sigma_y, obs.x, obs.y, lm.x, lm.y);
          std::cout << "multi_prob " << multi_prob << std::endl;
          particles[i].weight *= multi_prob;
        }
      }
    }

    // Sum up weights for normalization
    weight_sum += particles[i].weight;
  }

  weights.clear();
  //Normalize weights
  for (int j=0; j<num_particles; ++j){
    particles[j].weight /= weight_sum;
    weights.push_back(particles[j].weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> new_particles;

  std::default_random_engine gen;
  std::discrete_distribution<> d(weights.begin(), weights.end());

  for (int i=0; i<num_particles; ++i){
    std::cout << d(gen);
    int index = d(gen);
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
