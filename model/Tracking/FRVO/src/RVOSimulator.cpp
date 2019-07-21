/*
 *  RVOSimulator.cpp
 *  RVO2 Library.
 *  
 *  
 *  Copyright (C) 2008-10 University of North Carolina at Chapel Hill.
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
 *  North Carolina at Chapel Hill. The software program and documentation are
 *  supplied "as is", without any accompanying services from the University of
 *  North Carolina at Chapel Hill or the authors. The University of North
 *  Carolina at Chapel Hill and the authors do not warrant that the operation of
 *  the program will be uninterrupted or error-free. The end-user understands
 *  that the program was developed for research purposes and is advised not to
 *  rely exclusively on the program for any reason.
 *  
 *  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR ITS
 *  EMPLOYEES OR THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 *  SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 *  ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE
 *  UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR THE AUTHORS HAVE BEEN ADVISED
 *  OF THE POSSIBILITY OF SUCH DAMAGE.
 *  
 *  THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND THE AUTHORS SPECIFICALLY
 *  DISCLAIM ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND ANY
 *  STATUTORY WARRANTY OF NON-INFRINGEMENT. THE SOFTWARE PROVIDED HEREUNDER IS
 *  ON AN "AS IS" BASIS, AND THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND
 *  THE AUTHORS HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 *  ENHANCEMENTS, OR MODIFICATIONS.
 *  
 *  Please send all BUG REPORTS to:
 *  
 *  geom@cs.unc.edu
 *  
 *  The authors may be contacted via:
 *  
 *  Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, and
 *  Dinesh Manocha
 *  Dept. of Computer Science
 *  Frederick P. Brooks Jr. Computer Science Bldg.
 *  3175 University of N.C.
 *  Chapel Hill, N.C. 27599-3175
 *  United States of America
 *  
 *  http://gamma.cs.unc.edu/RVO2/
 *  
 */

#include "RVOSimulator.h"

#include "Agent.h"
#include "KdTree.h"
#include "Obstacle.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if HAVE_OPENMP || _OPENMP
#include <omp.h>
#endif

namespace RVO
{
  RVOSimulator::RVOSimulator() : agents_(), defaultAgent_(0), globalTime_(0.0f), kdTree_(0), obstacles_(), timeStep_(1.0f)
  {
    kdTree_ = new KdTree(this);
  }

  RVOSimulator::RVOSimulator(float timeStep, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2& velocity) : agents_(), defaultAgent_(0), globalTime_(0.0f), kdTree_(0), obstacles_(), timeStep_(timeStep)
  {
    kdTree_ = new KdTree(this);
    defaultAgent_ = new Agent(this);

    defaultAgent_->maxNeighbors_ = maxNeighbors;
    defaultAgent_->maxSpeed_ = maxSpeed;
    defaultAgent_->neighborDist_ = neighborDist;
    defaultAgent_->radius_ = radius;
    defaultAgent_->timeHorizon_ = timeHorizon;
    defaultAgent_->timeHorizonObst_ = timeHorizonObst;
    defaultAgent_->velocity_ = velocity;
  }

  RVOSimulator::~RVOSimulator()
  {
    if (defaultAgent_ != 0) {
      delete defaultAgent_;
    }

    for (size_t i = 0; i < agents_.size(); ++i) {
      delete agents_[i];
    }

    for (size_t i = 0; i < obstacles_.size(); ++i) {
      delete obstacles_[i];
    }

    delete kdTree_;
  }

  size_t RVOSimulator::getAgentNumAgentNeighbors(size_t agentNo) const
  {
    return agents_[agentNo]->agentNeighbors_.size();
  }

  size_t RVOSimulator::getAgentAgentNeighbor(size_t agentNo, size_t neighborNo) const
  {
    return agents_[agentNo]->agentNeighbors_[neighborNo].second->id_;
  }

  size_t RVOSimulator::getAgentObstacleNeighbor(size_t agentNo, size_t neighborNo) const
  {
    return agents_[agentNo]->obstacleNeighbors_[neighborNo].second->id_;
  }

  size_t RVOSimulator::getAgentNumObstacleNeighbors(size_t agentNo) const
  {
    return agents_[agentNo]->obstacleNeighbors_.size();
  }

  size_t RVOSimulator::getAgentNumORCALines(size_t agentNo) const
  {
    return agents_[agentNo]->orcaLines_.size();
  }

  const Line& RVOSimulator::getAgentORCALine(size_t agentNo, size_t lineNo) const
  {
    return agents_[agentNo]->orcaLines_[lineNo];
  }

  size_t RVOSimulator::addAgent(const Vector2& position)
  {
    if (defaultAgent_ == 0) {
      return RVO_ERROR;
    }

    Agent* agent = new Agent(this);

    agent->position_ = position;
    agent->maxNeighbors_ = defaultAgent_->maxNeighbors_;
    agent->maxSpeed_ = defaultAgent_->maxSpeed_;
    agent->neighborDist_ = defaultAgent_->neighborDist_;
    agent->radius_ = defaultAgent_->radius_;
    agent->timeHorizon_ = defaultAgent_->timeHorizon_;
    agent->timeHorizonObst_ = defaultAgent_->timeHorizonObst_;
    agent->velocity_ = defaultAgent_->velocity_;

    agent->id_ = agents_.size();

    agents_.push_back(agent);

    return agents_.size() - 1;
  }

  size_t RVOSimulator::addAgent(const Vector2& position, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2& velocity)
  {
    Agent* agent = new Agent(this);
    
    agent->position_ = position;
    agent->maxNeighbors_ = maxNeighbors;
    agent->maxSpeed_ = maxSpeed;
    agent->neighborDist_ = neighborDist;
    agent->radius_ = radius;
    agent->timeHorizon_ = timeHorizon;
    agent->timeHorizonObst_ = timeHorizonObst;
    agent->velocity_ = velocity;

    agent->id_ = agents_.size();
    
    agents_.push_back(agent);

    return agents_.size() - 1;
  }

  size_t RVOSimulator::addObstacle(const std::vector<Vector2>& vertices)
  {
    if (vertices.size() < 2) {
      return RVO_ERROR;
    }

    size_t obstacleNo = obstacles_.size();

    for (size_t i = 0; i < vertices.size(); ++i) {
      Obstacle* obstacle = new Obstacle();
      obstacle->point_ = vertices[i];
      if (i != 0) {
        obstacle->prevObstacle = obstacles_.back();
        obstacle->prevObstacle->nextObstacle = obstacle;
      }
      if (i == vertices.size() - 1) {
        obstacle->nextObstacle = obstacles_[obstacleNo];
        obstacle->nextObstacle->prevObstacle = obstacle;
      } 
      obstacle->unitDir_ = normalize(vertices[(i == vertices.size() - 1 ? 0 : i+1)] - vertices[i]);

      if (vertices.size() == 2) {
        obstacle->isConvex_ = true;
      } else {
        obstacle->isConvex_ = (leftOf(vertices[(i == 0 ? vertices.size() - 1 : i-1)], vertices[i], vertices[(i == vertices.size() - 1 ? 0 : i+1)]) >= 0);
      }

      obstacle->id_ = obstacles_.size();

      obstacles_.push_back(obstacle);
    }

    return obstacleNo;
  }

  void RVOSimulator::doStep()
  {
    kdTree_->buildAgentTree();

#pragma omp parallel for

    for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
      agents_[i]->computeNeighbors();
      agents_[i]->computeNewVelocity();
    }

#pragma omp parallel for

    for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
      agents_[i]->update();
    }

    globalTime_ += timeStep_;
  }

  size_t RVOSimulator::getAgentMaxNeighbors(size_t agentNo) const
  {
    return agents_[agentNo]->maxNeighbors_;
  }

  float RVOSimulator::getAgentMaxSpeed(size_t agentNo) const
  {
    return agents_[agentNo]->maxSpeed_;
  }

  float RVOSimulator::getAgentNeighborDist(size_t agentNo) const
  {
    return agents_[agentNo]->neighborDist_;
  }

  const Vector2& RVOSimulator::getAgentPosition(size_t agentNo) const
  {
    return agents_[agentNo]->position_;
  }

  const Vector2& RVOSimulator::getAgentPrefVelocity(size_t agentNo) const
  {
    return agents_[agentNo]->prefVelocity_;
  }

  float RVOSimulator::getAgentRadius(size_t agentNo) const
  {
    return agents_[agentNo]->radius_;
  }

  float RVOSimulator::getAgentTimeHorizon(size_t agentNo) const
  {
    return agents_[agentNo]->timeHorizon_;
  }

  float RVOSimulator::getAgentTimeHorizonObst(size_t agentNo) const
  {
    return agents_[agentNo]->timeHorizonObst_;
  }

  const Vector2& RVOSimulator::getAgentVelocity(size_t agentNo) const
  {
    return agents_[agentNo]->velocity_;
  }

  float RVOSimulator::getGlobalTime() const
  {
    return globalTime_;
  }

  size_t RVOSimulator::getNumAgents() const
  {
    return agents_.size();
  }

  size_t RVOSimulator::getNumObstacleVertices() const
  {
    return obstacles_.size();
  }

  const Vector2& RVOSimulator::getObstacleVertex(size_t vertexNo) const
  {
    return obstacles_[vertexNo]->point_;
  }

  size_t RVOSimulator::getNextObstacleVertexNo(size_t vertexNo) const
  {
    return obstacles_[vertexNo]->nextObstacle->id_;
  }

  size_t RVOSimulator::getPrevObstacleVertexNo(size_t vertexNo) const
  {
    return obstacles_[vertexNo]->prevObstacle->id_;
  }

  float RVOSimulator::getTimeStep() const
  {
    return timeStep_;
  }

  void RVOSimulator::processObstacles()
  {
    kdTree_->buildObstacleTree();
  }

  bool RVOSimulator::queryVisibility(const Vector2& point1, const Vector2& point2, float radius) const
  {
    return kdTree_->queryVisibility(point1, point2, radius);
  }
  
  void RVOSimulator::setAgentDefaults(float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2& velocity)
  {
    if (defaultAgent_ == 0) {
      defaultAgent_ = new Agent(this);
    }
    
    defaultAgent_->maxNeighbors_ = maxNeighbors;
    defaultAgent_->maxSpeed_ = maxSpeed;
    defaultAgent_->neighborDist_ = neighborDist;
    defaultAgent_->radius_ = radius;
    defaultAgent_->timeHorizon_ = timeHorizon;
    defaultAgent_->timeHorizonObst_ = timeHorizonObst;
    defaultAgent_->velocity_ = velocity;
  }

  void RVOSimulator::setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors)
  {
    agents_[agentNo]->maxNeighbors_ = maxNeighbors;
  }

  void RVOSimulator::setAgentMaxSpeed(size_t agentNo, float maxSpeed)
  {
    agents_[agentNo]->maxSpeed_ = maxSpeed;
  }

  void RVOSimulator::setAgentNeighborDist(size_t agentNo, float neighborDist)
  {
    agents_[agentNo]->neighborDist_ = neighborDist;
  }

  void RVOSimulator::setAgentPosition(size_t agentNo, const Vector2& position)
  {
    agents_[agentNo]->position_ = position;
  }

  void RVOSimulator::setAgentPrefVelocity(size_t agentNo, const Vector2& prefVelocity)
  {
    agents_[agentNo]->prefVelocity_ = prefVelocity;
  }

  void RVOSimulator::setAgentRadius(size_t agentNo, float radius)
  {
    agents_[agentNo]->radius_ = radius;
  }

  void RVOSimulator::setAgentTimeHorizon(size_t agentNo, float timeHorizon)
  {
    agents_[agentNo]->timeHorizon_ = timeHorizon;
  }

  void RVOSimulator::setAgentTimeHorizonObst(size_t agentNo, float timeHorizonObst)
  {
    agents_[agentNo]->timeHorizonObst_ = timeHorizonObst;
  }

  void RVOSimulator::setAgentVelocity(size_t agentNo, const Vector2& velocity)
  {
    agents_[agentNo]->velocity_ = velocity;
  }

  void RVOSimulator::setTimeStep(float timeStep)
  {
    timeStep_ = timeStep;
  }
}
