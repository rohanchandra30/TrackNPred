/*
 *  FRVO
 *  Modified using RVO2 Library.
 *  
 *  
 *  Copyright (C) 2018 University of Maryland, College Park
 *  All rights reserved.
 *  
 *  Permission to use, copy, modify, and distribute this software and its
 *  documentation for educational, research, and non-profit purposes, without
 *  fee, and without a written agreement is hereby granted, provided that the
 *  above copyright notice, this paragraph, and the following four paragraphs
 *  appear in all copies.
 *  
 *  Permission to incorporate this software into commercial products may be
 *  obtained by contacting the University of North Carolina at Chapel Hill.
 *  
 *  This software program and documentation are copyrighted by the University of
 *  Maryland, College Park. The software program and documentation are
 *  supplied "as is", without any accompanying services from the University of
 *  Maryland, College Park or the authors. The University of Maryland, College Park
 *  and the authors do not warrant that the operation of
 *  the program will be uninterrupted or error-free. The end-user understands
 *  that the program was developed for research purposes and is advised not to
 *  rely exclusively on the program for any reason.
 *  
 *  IN NO EVENT SHALL THE UNIVERSITY OF MARYLAND, COLLEGE PARK OR ITS
 *  EMPLOYEES OR THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 *  SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 *  ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE
 *  UNIVERSITY OF MARYLAND, COLLEGE PARK OR THE AUTHORS HAVE BEEN ADVISED
 *  OF THE POSSIBILITY OF SUCH DAMAGE.
 *  
 *  THE UNIVERSITY OF MARYLAND, COLLEGE PARK AND THE AUTHORS SPECIFICALLY
 *  DISCLAIM ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND ANY
 *  STATUTORY WARRANTY OF NON-INFRINGEMENT. THE SOFTWARE PROVIDED HEREUNDER IS
 *  ON AN "AS IS" BASIS, AND THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND
 *  THE AUTHORS HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 *  ENHANCEMENTS, OR MODIFICATIONS.
 *  
 *  Please send all BUG REPORTS to:
 *  
 *  rohan@cs.umd.edu, uttaranb@cs.umd.edu 
 *  
 */

/*
 * This file implements the Front-RVO version of RVO where it predicts the positions of agents in a dense crowd.
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#if HAVE_OPENMP || _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <RVO/RVO.h>
#else
#include "RVO.h"
#endif

using namespace std;

/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;


/*---------------------------------------------------------------------------------------*/

void setupScenario(RVO::RVOSimulator* sim, const char* filename)
{
    /* Specify the global time step of the simulation. */
    sim->setTimeStep(0.25f);

    /* Specify the default parameters for agents that are subsequently added. */
    sim->setAgentDefaults(15.0f, 10, 10.0f, 10.0f, 1.5f, 2.0f);
//    sim->setAgentDefaults(2.0f, 10, 0.01f, 10.0f, 0.2f, 10.0f);

    /*
    * Add agents, specifying their start position, and store their goals on the
    * opposite side of the environment.
    */
//  for (int i = 0; i < 250; ++i) {
//    sim->addAgent(200.0f *
//      RVO::Vector2(std::cos(i * 2.0f * M_PI / 250.0f),
//      std::sin(i * 2.0f * M_PI / 250.0f)));
//    goals.push_back(-sim->getAgentPosition(i));
//  }
    ifstream posFile(filename);
    string line;
    if (posFile.is_open()) {
        while(getline(posFile, line)) {
            size_t i = 0;
            double pos[4], scale = 1; // pos[0] = x, pos[1] = y, pos[2] = vx, pos[3] = vy
            char* line_cstr = new char[line.length() + 1];
            char* pEnd;
            strcpy(line_cstr, line.c_str());
//            char *token = std::strtok(cstr, " ");
//            delete [] cstr;
//            while (token != NULL) {
//                const char* t = token;
//                pos[i++] = atof(t);
//                token = std::strtok(NULL, " ");
//            }
            pos[0] = strtod(line_cstr, &pEnd);
            pos[1] = strtod(pEnd, &pEnd);
            pos[2] = strtod(pEnd, &pEnd);
            pos[3] = strtod (pEnd, NULL);
            sim->addAgent(RVO::Vector2(pos[0], pos[1]));
//            goals.push_back(RVO::Vector2(pos[0]+scale*pos[2], pos[1]+scale*pos[3]));
	    goals.push_back(-sim->getAgentPosition(i));
        }
        posFile.close();
    }
}

/*---------------------------------------------------------------------------------------*/

void updateVisualization(RVO::RVOSimulator* sim, const char* filename)
{
  /* Output the current global time. */
//  outfile << sim->getGlobalTime() << " ";

  /* Output the current position of all the agents. */
  ofstream posFile(filename, std::ofstream::out);
  if (posFile.is_open()) {
      for (size_t i = 0; i < sim->getNumAgents(); ++i)
          posFile << sim->getAgentPosition(i).x() << " " << sim->getAgentPosition(i).y() << "\n";
      posFile.close();
  }
}

void setPreferredVelocities(RVO::RVOSimulator* sim)
{
  /* 
   * Set the preferred velocity to be a vector of unit magnitude (speed) in the
   * direction of the goal.
   */
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(sim->getNumAgents()); ++i) {
    RVO::Vector2 goalVector = goals[i] - sim->getAgentPosition(i);
    if (RVO::absSq(goalVector) > 1.0f) {
      goalVector = RVO::normalize(goalVector);
    }
    sim->setAgentPrefVelocity(i, goalVector);
  }
}

bool reachedGoal(RVO::RVOSimulator* sim)
{
  /* Check if all agents have reached their goals. */
  for (size_t i = 0; i < sim->getNumAgents(); ++i) {
    if (RVO::absSq(sim->getAgentPosition(i) - goals[i]) > sim->getAgentRadius(i) * sim->getAgentRadius(i)) {
      return false;
    }
  }

  return true;
}

int main(int argc, const char* argv[])
{
    /* Create a new simulator instance. */
    RVO::RVOSimulator* sim = new RVO::RVOSimulator();
    argv[1] = "frvoPos.txt";
    argv[2] = "frvoPosPred.txt";

    /* Set up the scenario. */
    setupScenario(sim, argv[1]);

    /* Perform (and manipulate) the simulation. */
    int numSteps = 1;
    for (size_t i = 0; i < numSteps && !reachedGoal(sim); i++) {
        setPreferredVelocities(sim);
        sim->doStep();
    }
    updateVisualization(sim, argv[2]);

    delete sim;

    return 0;
}
