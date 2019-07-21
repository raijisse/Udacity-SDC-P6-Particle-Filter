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
#include <cmath>

#include "helper_functions.h"

using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for all variables
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 100; // TODO: Set the number of particles

  particles.resize(num_particles);
  weights.resize(num_particles);

  for (int i = 0; i < num_particles; ++i)
  {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
    weights[i] = 1.0;
  }

  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++)
  {
    double x0, y0, theta0, xf, yf, thetaf;
    x0 = particles[i].x;
    y0 = particles[i].y;
    theta0 = particles[i].theta;

    if (fabs(yaw_rate) > 0.0000001)
    {
      xf = x0 + velocity / yaw_rate * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
      yf = y0 + velocity / yaw_rate * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
      thetaf = theta0 + yaw_rate * delta_t;
    }
    else
    {
      xf = x0 + velocity * delta_t * cos(theta0);
      yf = y0 + velocity * delta_t * sin(theta0);
      thetaf = theta0;
    }

    std::normal_distribution<double> noisy_x(xf, std_pos[0]);
    std::normal_distribution<double> noisy_y(yf, std_pos[1]);
    std::normal_distribution<double> noisy_theta(thetaf, std_pos[2]);

    particles[i].x = noisy_x(gen);
    particles[i].y = noisy_y(gen);
    particles[i].theta = noisy_theta(gen);
  }
}



void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned i = 0; i < observations.size(); i++)
  {
    double minimal_dist = 10000;

    for (unsigned j = 0; j < predicted.size(); j++)
    {
      // init minimal distance with first object
      // An other solution would be to remove this if block and initialize it with some large number
      if (dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y) < minimal_dist)
      {
        minimal_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
        observations[i].id = predicted[j].id;
      }
    }
  }
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
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  for (int i = 0; i < num_particles; i++)
  {
    double particle_x, particle_y, particle_theta;
    particle_x = particles[i].x;
    particle_y = particles[i].y;
    particle_theta = particles[i].theta;
    // reinitialize weights
    particles[i].weight = 1.0;

    // Convert coordinates:
    vector<LandmarkObs> converted_obs;
    converted_obs.resize(observations.size());

    for (unsigned j = 0; j < observations.size(); j++)
    {
      double x_map, y_map;
      x_map = particle_x + observations[j].x * cos(particle_theta) - observations[j].y * sin(particle_theta);
      y_map = particle_y + observations[j].x * sin(particle_theta) + observations[j].y * cos(particle_theta);

      converted_obs[j].id = -1;
      converted_obs[j].x = x_map;
      converted_obs[j].y = y_map;
    }

    // Keep landmarks in sensor range
    vector<LandmarkObs> reachable_landmarks;
    for (unsigned k = 0; k < map_landmarks.landmark_list.size(); k++)
    {
      double dist_to_landmark;
      dist_to_landmark = dist(particle_x, particle_y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);

      if (dist_to_landmark <= sensor_range)
      {
        LandmarkObs to_append;
        to_append.id = map_landmarks.landmark_list[k].id_i;
        to_append.x = map_landmarks.landmark_list[k].x_f;
        to_append.y = map_landmarks.landmark_list[k].y_f;
        reachable_landmarks.push_back(to_append);
      }
    }

    // Associate landmarks with observations
    dataAssociation(reachable_landmarks, converted_obs);

    // For each converted obs associated, update weight of the particle
    for (unsigned l = 0; l < converted_obs.size(); l++)
    {
      double x_obs, y_obs, mu_x, mu_y;
      x_obs = converted_obs[l].x;
      y_obs = converted_obs[l].y;
      if (reachable_landmarks.size() > 0)
      {
        for (unsigned m = 0; m < reachable_landmarks.size(); m++)
        {
          if (converted_obs[l].id == reachable_landmarks[m].id)
          {
            mu_x = reachable_landmarks[m].x;
            mu_y = reachable_landmarks[m].y;
            particles[i].weight *= multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
          }
        }
      }
      else
      {
        particles[i].weight = 0;
      }
      weights[i] = particles[i].weight;
    }
  }
}


void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  std::discrete_distribution<> dist(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles;

  resampled_particles.resize(num_particles);

  for (int i = 0; i < num_particles; ++i)
  {
    resampled_particles[i] = particles[dist(gen)];
  }
  particles = resampled_particles;
}


void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}