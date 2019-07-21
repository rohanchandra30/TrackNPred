// BayesianFilter2.cpp : Defines the entry point for the console application.
//

//#include <tchar.h>
#include "../../KalmanFilter/bayesianFilter.h"

#include "RVO.h"

//#include "cVector2.h"
typedef RVO::Vector2 cVector2;
int Peds = 1;

struct PedData
{
	public:
		cVector2* PData;
		PedData()
		{
			PData= new cVector2[Peds];
		}
};

#include <vector>
#include <fstream>
#include <iomanip>


using namespace kf;

int CallCount = 0;

#define DYNAMIC_STEP 0	//only can learn
float sampling = 6;
unsigned int numLearning =5;

bool bUseLastFrameGoal = false;

float dataNoise = 15/100.0;	//centimeter to meter

//flags
bool bDrawGoal = true;
bool bDrawTrajectory = false;
bool bAnimationOn = false;
bool bfmeanEach = true;
bool bOutputConstModel = true;
const bool bUseRVOModel = true;	//always use 
bool bUseEM = true;

#define X_DIM 6  // State dimension
#define U_DIM 6  // Control Input dimension
#define Z_DIM 2  // Measurement dimension -  position:(x,y)
#define M_DIM 6  // Motion Noise dimension
#define N_DIM 2  // Measurement Noise dimension

float initialSigma[X_DIM] = {0.5 /*posx*/, 0.5/*posy*/, 0.5/*velx*/, 0.5/*vely*/, 0.5/*goalx*/, 0.5/*goaly*/};
float processNoise[X_DIM] = {0.5 /*posx*/, 0.5/*posy*/, 0.5/*velx*/, 0.5/*vely*/, 0.5/*goalx*/, 0.5/*goaly*/};

float initialGuessRadius = 0.2;
float initialGuessPrefSpeed = 1.4;
float initialGuessTTC = 0.4;

enum StateIndex{POS_X=0, POS_Y, VEL_X, VEL_Y, GOAL_X, GOAL_Y};

//view
int cx=512, cy=512;
float eye[3];
float dest[3];

RVO::RVOSimulator* sim = NULL;

int numAgt = 1;
int g_numFrames = 10;
unsigned int g_startFrame=0;
unsigned int g_endFrame = 0;
unsigned int g_curFrame = 0;	//original value
unsigned int g_curAgt = 0;

// pop-in, out
unsigned int startAgt = 0;
unsigned int endAgt = 0;

#define INF 1e6
#define EPS 1e-6
#define INVALID 0xffffffff
RVO::Vector2 vINVALID(INVALID, INVALID);
RVO::Vector2 vZERO(0, 0);

//agent
#define MAX_AGENT 1000
#define NUM_SAMPLES 1000

int g_step = 0;
double g_dt = 0.4;
float fps = 0;

unsigned int learningCnt = 0;

unsigned int agentID[MAX_AGENT];
unsigned int startFrame[MAX_AGENT];
unsigned int endFrame[MAX_AGENT];
std::vector<RVO::Vector2> position[MAX_AGENT];
std::vector<RVO::Vector2> velocity[MAX_AGENT];
std::vector<RVO::Vector2> acceleration[MAX_AGENT];

//occlusion
bool bVisible[MAX_AGENT];

typedef Matrix<X_DIM> stateVector;
float abs(stateVector v)
{
	float sum = 0;
	for(int i=0; i<X_DIM; i++)
	{
		sum += (v[i]*v[i]);
	}

	return sqrt(sum);
}

//EM - when using fmean
stateVector states[MAX_AGENT];
stateVector prevStates[MAX_AGENT];
stateVector predStates[MAX_AGENT];

struct matKF
{
	Matrix<X_DIM,X_DIM> Sigma;
	Matrix<X_DIM> xHat;
	Matrix<Z_DIM> z;
	Matrix<U_DIM> u;
	std::vector<Matrix<X_DIM> > X;
} ;

//Individual EM
std::vector<Matrix<X_DIM,X_DIM>> Mprev;
std::vector<Matrix<X_DIM,X_DIM>> M;
float diffScore;

//newEM
std::vector<stateVector > Xd[MAX_AGENT];
std::vector<stateVector > prevXd[MAX_AGENT];
std::vector<stateVector > predXd[MAX_AGENT];

std::vector<matKF> mat;
std::vector<Matrix<X_DIM> > ensemble(NUM_SAMPLES);

std::ofstream fout;		//result - filtered states & distribution
std::ofstream fout2;	//result - position
std::ofstream fout3;	//result - predicted states & distribution (fmean)



//kf
Matrix<X_DIM> f_PV(const Matrix<X_DIM>& x, const Matrix<U_DIM>& u, const Matrix<M_DIM>& m);
Matrix<X_DIM> f_RVO(const Matrix<X_DIM>& x, const Matrix<U_DIM>& u, const Matrix<M_DIM>& m);
Matrix<X_DIM> dynamic_f(const Matrix<X_DIM>& x, const Matrix<U_DIM>& u, const Matrix<M_DIM>& m);
Matrix<Z_DIM> h(const Matrix<X_DIM>& x, const Matrix<N_DIM>& n);

//use first 2 frames to initialize...
void initKF()
{
	unsigned int curFrame = 0;

	//multiple Agents
	for (int i=0; i<numAgt; i++)
	{
		//init matrices
		matKF tempMat;
		tempMat.Sigma = kf::identity<X_DIM>();
		tempMat.xHat = zeros<X_DIM>();

		tempMat.z = zeros<Z_DIM>();
		tempMat.u = zeros<U_DIM>();

		//sim->setAgentDefaults(15.0f, 10, 10.0f, 10.0f, 1.5f, 2.0f);
		//initial guess - default parameters
		tempMat.xHat[POS_X] = position[i][curFrame].x();		//posX
		tempMat.xHat[POS_Y] = position[i][curFrame].y();		//posY

		tempMat.xHat[VEL_X] = (position[i][1].x()-position[i][0].x())/g_dt;		//velX
		tempMat.xHat[VEL_Y] = (position[i][1].y()-position[i][0].y())/g_dt;		//velY

		if( bUseLastFrameGoal )
		{
			tempMat.xHat[GOAL_X] = (position[i][endFrame[i]-startFrame[i]].x()-position[i][0].x())/(g_dt*(endFrame[i]-startFrame[i]));
			tempMat.xHat[GOAL_Y] = (position[i][endFrame[i]-startFrame[i]].y()-position[i][0].y())/(g_dt*(endFrame[i]-startFrame[i]));
		}
		else 
		{
			tempMat.xHat[GOAL_X] = (position[i][1].x()-position[i][0].x())/g_dt;		//velX
			tempMat.xHat[GOAL_Y] = (position[i][1].y()-position[i][0].y())/g_dt;		//velY
		}

		for(int s = 0; s<X_DIM; s++)
		{
			tempMat.Sigma(s,s) = initialSigma[s]; //confidence
		}

		tempMat.X.resize(NUM_SAMPLES);
//		std::copy(ensemble.begin(),ensemble.end(),tempMat.X.begin());

		for (size_t j = 0; j < tempMat.X.size(); ++j) {
			tempMat.X[j] = sampleGaussian(tempMat.xHat, tempMat.Sigma);
			//fout<<g_curFrame-g_startFrame<<" "<<i<<" "<<j<<" "<<~tempMat.X[j];
		}
		mat.push_back(tempMat);

		//copy to state backup
		states[i] = tempMat.xHat;
		prevStates[i] = states[i];

		//all samples
		predXd[i] = prevXd[i] = Xd[i] = tempMat.X;

		//init M;
		Matrix<X_DIM> mProcessNoise;
		for (int i=0; i<X_DIM; i++) {
			mProcessNoise[i] = processNoise[i];
		}

		M.push_back(mProcessNoise * ~mProcessNoise);
		Mprev.push_back(mProcessNoise * ~mProcessNoise);
		//Mprev = kf::identity<X_DIM>();
	}
}

void fmeanEach_PV()
{
	//PV without noise
	for(int i=0; i<numAgt; i++)
	{
		//mean distribution
		predStates[i] = zeros<X_DIM>();

		for(int s=0; s<NUM_SAMPLES; s++)
		{
			//p from v
			predXd[i][s][POS_X] = prevXd[i][s][POS_X] + prevXd[i][s][VEL_X]*g_dt;
			predXd[i][s][POS_Y] = prevXd[i][s][POS_Y] + prevXd[i][s][VEL_Y]*g_dt;
			
			//v
			predXd[i][s][VEL_X] = prevXd[i][s][VEL_X];
			predXd[i][s][VEL_Y] = prevXd[i][s][VEL_Y];

			//goal
			predXd[i][s][GOAL_X] = prevXd[i][s][VEL_X];
			predXd[i][s][GOAL_Y] = prevXd[i][s][VEL_Y];

			//mean distribution
			predStates[i] += predXd[i][s] / NUM_SAMPLES;
		}
	}
}

void fmeanEach_RVO()
{
	for (int i = 0; i < numAgt; i++)
	{
		predStates[i] = zeros<X_DIM>();

		for ( int s = 0; s<NUM_SAMPLES; s++)
		{
			//reset others - prevStates
			for (int neighbor = 0; neighbor < numAgt; neighbor++)
			{
				cVector2 pos(states[neighbor][POS_X], states[neighbor][POS_Y]);
				cVector2 curVel(states[neighbor][VEL_X], states[neighbor][VEL_Y]);
				cVector2 prefVel(0,0);

				if(bUseLastFrameGoal)
				{
					float prefSpeed = RVO::abs(curVel);
					prefVel = prefSpeed * RVO::normalize(position[neighbor][endFrame[neighbor]-startFrame[neighbor]]-pos);
				}
				else
				{
					//prefVel = curVel;
					prefVel = cVector2(states[neighbor][GOAL_X], states[neighbor][GOAL_Y]);
				}

				sim->setAgentPosition(agentID[neighbor], pos);
				sim->setAgentVelocity(agentID[neighbor], curVel);
				sim->setAgentPrefVelocity(agentID[neighbor], prefVel);
			}

			// update for sample
			{
				cVector2 pos(prevXd[i][s][POS_X], prevXd[i][s][POS_Y]);
				cVector2 curVel(prevXd[i][s][VEL_X], prevXd[i][s][VEL_Y]);
				cVector2 prefVel(0,0);

				if(bUseLastFrameGoal)
				{
					float prefSpeed = RVO::abs(curVel);
					prefVel = prefSpeed * RVO::normalize(position[i][endFrame[i]-startFrame[i]]-pos);
				}
				else
				{
					//prefVel = curVel;
					prefVel = cVector2(prevXd[i][s][GOAL_X], prevXd[i][s][GOAL_Y]);
				}

				sim->setAgentPosition(agentID[i], pos);
				sim->setAgentVelocity(agentID[i], curVel);
				sim->setAgentPrefVelocity(agentID[i], prefVel);
			}

			sim->doStep();

			//copy back to sample
			cVector2 pos = sim->getAgentPosition(agentID[i]);
			cVector2 vel(prevXd[i][s][VEL_X], prevXd[i][s][VEL_Y]);
			//cVector2 vel = sim->getAgentVelocity(agentID[i]);
			cVector2 goal(prevXd[i][s][GOAL_X], prevXd[i][s][GOAL_Y]);			

			predXd[i][s][POS_X]		= pos.x();
			predXd[i][s][POS_Y]		= pos.y();
			predXd[i][s][VEL_X]		= vel.x();
			predXd[i][s][VEL_Y]		= vel.y();
			predXd[i][s][GOAL_X]	= goal.x();
			predXd[i][s][GOAL_Y]	= goal.y();

			//mean distribution
			predStates[i] += predXd[i][s] / NUM_SAMPLES;
		}
	}
}



//predict - RVO Step
PedData predict()
{
	if(g_curFrame >= g_endFrame)
	{
		std::cout<<"no more data "<<std::endl;
		bAnimationOn = false;
		//return;
	}

	std::cout<<"predict g_step: "<<g_step<<std::endl;

	//update RVO states
	int curFrameIdx = g_curFrame - g_startFrame;

	for(int i=0; i<numAgt; i++)
	//for(int i=numAgt-1; i>=0; i--)
	{
		cVector2 pos(states[i][POS_X], states[i][POS_Y]);
		cVector2 curVel(states[i][VEL_X], states[i][VEL_Y]);
		cVector2 prefVel(0,0);

		//use vel magnitude
		float prefSpeed = RVO::abs(curVel);

		if(bUseLastFrameGoal)
		{
			prefVel = prefSpeed * RVO::normalize(position[i][endFrame[i]-startFrame[i]]-pos);
		}
		else	//extend curVel - goal position after one second
		{
			//from last frame position
			//cVector2 immediateGoalPos = pos + curVel * g_dt;
			//cVector2 goalPos = position[i][0] + curVel
			//prefVel = prefSpeed * normalize(curVel);
			
			//prefVel = curVel;
			prefVel = cVector2(states[i][GOAL_X], states[i][GOAL_Y]);
		}

		sim->setAgentPosition(agentID[i],pos);
		sim->setAgentVelocity(agentID[i],curVel);
		sim->setAgentPrefVelocity(agentID[i],prefVel);
	}

	sim->doStep();
	
	//get predicted results
	PedData Output;
	for(int i=0; i<numAgt; i++)
	//for(int i=numAgt-1; i>=0; i--)
	{
		int localFrameIdx = g_curFrame - startFrame[i];

		//update preferred velocity
		cVector2 pos = sim->getAgentPosition(i);
		cVector2 vel = sim->getAgentVelocity(i);

		mat[i].xHat[POS_X] = pos.x();
		mat[i].xHat[POS_Y] = pos.y();
		mat[i].xHat[VEL_X] = vel.x();
		mat[i].xHat[VEL_Y] = vel.y();

		states[i] = mat[i].xHat;
		
		Output.PData[i] = RVO::Vector2(position[i][localFrameIdx+1].x(),position[i][localFrameIdx+1].y());
	}
	
	g_step++;
	g_curFrame++;
	return Output;
}

void constVelocity()
{
	std::cout<<"...constVelocity model... ";

	
	unsigned int lastFrame = numLearning;
	if( lastFrame >= g_numFrames)
	{
		lastFrame = g_numFrames-1;
	}

	//const-vel model
	cVector2 lastPos[MAX_AGENT];
	cVector2 lastVel[MAX_AGENT];

	for(int i=0; i<numAgt; i++)
	{
		lastPos[i] = position[i][lastFrame];
		lastVel[i] = (position[i][lastFrame] - position[i][lastFrame-1])/g_dt;
	}

	
	std::cout<<"done"<<std::endl;
}

void constAcceleration()
{
	std::cout<<"...constAcceleration model... ";


	unsigned int lastFrame = numLearning;
	if( lastFrame >= g_numFrames)
	{
		lastFrame = g_numFrames-1;
	}

	//const-vel model
	cVector2 lastPos[MAX_AGENT];
	cVector2 lastVel[MAX_AGENT];
	cVector2 lastAcc[MAX_AGENT];

	for(int i=0; i<numAgt; i++)
	{
		lastPos[i] = position[i][lastFrame];
		lastVel[i] = (position[i][lastFrame] - position[i][lastFrame-1])/g_dt;

		if( lastFrame > 1 )
		{
			lastAcc[i] = (position[i][lastFrame] - 2 * position[i][lastFrame-1] + position[i][lastFrame - 2])/(g_dt*g_dt);
		}
		else
		{
			lastAcc[i] = cVector2(0,0);
		}
	}

	
	std::cout<<"done"<<std::endl;
}




Matrix<X_DIM> f_PV(const Matrix<X_DIM>& x, const Matrix<U_DIM>& u, const Matrix<M_DIM>& m) {


	assert(x[0] == x[0]);

	//neighbor information
	for (int i = 0; i < numAgt; i++)
	{
		cVector2 pos(states[i][POS_X], states[i][POS_Y]);
		cVector2 curVel(states[i][VEL_X], states[i][VEL_Y]);
	}
	 
	//training - one agent
	cVector2 pos(x[0], x[1]);
	cVector2 curVel(x[2], x[3]);
	cVector2 prefVel(0,0);

	if(bUseLastFrameGoal)
	{
		//speed: dist left / numFrames
		prefVel = (position[g_curAgt][endFrame[g_curAgt]-startFrame[g_curAgt]]-pos)/(g_numFrames - g_curFrame);
	}
	else
	{
		prefVel = curVel;
	}

	Matrix<X_DIM> xNew; //Scale m!
	Matrix<M_DIM> mNew;

	if(bUseEM)
		mNew = sampleGaussian(zeros<M_DIM>(), Mprev[g_curAgt]);
	else
		mNew = m;

	xNew[0] = x[0] + x[2]*g_dt + mNew[0];
	xNew[1] = x[1] + x[3]*g_dt + mNew[1];

	xNew[2] = x[2] + mNew[2];
	xNew[3] = x[3] + mNew[3];

	xNew[4] = x[4] + mNew[4];
	xNew[5] = x[5] + mNew[5];

	return xNew;
}

Matrix<X_DIM> f_RVO(const Matrix<X_DIM>& x, const Matrix<U_DIM>& u, const Matrix<M_DIM>& m) {

	assert(x[0] == x[0]);

	//neighbor information
	for (int i = 0; i < numAgt; i++)
	{
		cVector2 pos(states[i][POS_X], states[i][POS_Y]);
		cVector2 curVel(states[i][VEL_X], states[i][VEL_Y]);
		cVector2 prefVel(0,0);

		//use vel magnitude
		float prefSpeed = RVO::abs(curVel);

		if(bUseLastFrameGoal)
		{
			prefVel = prefSpeed * RVO::normalize(position[i][endFrame[i]-startFrame[i]]-pos);
		}
		else	//extend curVel - goal position after one second
		{
			//from first frame positoin
			//cVector2 immediateGoalPos = pos + curVel * g_dt;
			//cVector2 goalPos = position[i][0] + curVel
			//prefVel = prefSpeed * normalize(curVel);
			
			//prefVel = curVel;
			prefVel = cVector2(states[i][GOAL_X], states[i][GOAL_Y]);
		}

		sim->setAgentPosition(agentID[i], pos);
		sim->setAgentVelocity(agentID[i], curVel);
		sim->setAgentPrefVelocity(agentID[i], prefVel);
	}
	 
	//training - one agent
	{
		cVector2 pos(x[0], x[1]);
		cVector2 curVel(x[2], x[3]);
		cVector2 prefVel(0,0);

		sim->setAgentPosition(agentID[g_curAgt], pos);
		sim->setAgentVelocity(agentID[g_curAgt], curVel);

		if(bUseLastFrameGoal)
		{
			//speed: dist left / numFrames
			prefVel = (position[g_curAgt][endFrame[g_curAgt]-startFrame[g_curAgt]]-pos)/(g_numFrames - g_curFrame);
		}
		else
		{
			//prefVel = curVel;
			prefVel = cVector2(x[GOAL_X], x[GOAL_Y]);
		}

		sim->setAgentPrefVelocity(agentID[g_curAgt], prefVel);	// #120116 extrapolate previous velocity
	}

	sim->doStep();

	Matrix<X_DIM> xNew; //Scale m!
	Matrix<M_DIM> mNew;

	if(bUseEM)
		mNew = sampleGaussian(zeros<M_DIM>(), Mprev[g_curAgt]);
	else
		mNew = m;

	xNew[0] = sim->getAgentPosition(agentID[g_curAgt]).x() + mNew[0];
	xNew[1] = sim->getAgentPosition(agentID[g_curAgt]).y() + mNew[1];

//	xNew[2] = sim->getAgentVelocity(agentID[g_curAgt]).x() + mNew[2];
//	xNew[3] = sim->getAgentVelocity(agentID[g_curAgt]).y() + mNew[3];

	xNew[2] = x[2] + mNew[2];
	xNew[3] = x[3] + mNew[3];

	xNew[4] = x[4] + mNew[4];
	xNew[5] = x[5] + mNew[5];

	return xNew;
}

// Sensor model
Matrix<Z_DIM> h(const Matrix<X_DIM>& x, const Matrix<N_DIM>& n) {
	Matrix<Z_DIM> z;
	for (unsigned a = 0; a < Z_DIM; a++){
		z[a] = x[a] + dataNoise*n[a];
	}

	return z;
}



void initRVO()
{
	sim = new RVO::RVOSimulator();
	sim->setTimeStep(g_dt);

	/* Specify the default parameters for agents that are subsequently added. */
	sim->setAgentDefaults(2.0f, 10, 0.01f, 10.0f, 0.2f, 10.0f);


	//visible test
	bool visible = true;

	for(int i=0; i<numAgt; i++)
	{
		bVisible[i] = visible;
//		visible = !visible;

		if( startFrame[i] <= g_curFrame && g_curFrame < endFrame[i] )
		{
			agentID[i] = sim->agents_.size();
			sim->addAgent(position[i][0]);
			std::cout<<"agent "<<i<<" added"<<std::endl;
		}
	}
}



void resetAll()
{
	//reset global variables
	g_step = 0;
	learningCnt = 0;
	g_curAgt = 0;
	g_curFrame = g_startFrame;
	unsigned int curFrame = 0;

	//resetRVO
	for(int i=0; i<numAgt; i++)
	{
		if( startFrame[i] <= g_curFrame && g_curFrame < endFrame[i] )
		{
			sim->setAgentPosition(agentID[i], position[i][0]);
		}
	}

	//multiple Agents
	for (int i=0; i<numAgt; i++)
	{
		//init matrices
		mat[i].Sigma = kf::identity<X_DIM>();
		mat[i].xHat = zeros<X_DIM>();

		mat[i].z = zeros<Z_DIM>();
		mat[i].u = zeros<U_DIM>();

		//sim->setAgentDefaults(15.0f, 10, 10.0f, 10.0f, 1.5f, 2.0f);
		//initial guess - default parameters
		mat[i].xHat[0] = position[i][curFrame].x();		//posX
		mat[i].xHat[1] = position[i][curFrame].y();		//posY

		if( bUseLastFrameGoal )
		{
			mat[i].xHat[2] = (position[i][endFrame[i]-startFrame[i]].x()-position[i][0].x())/(g_dt*(endFrame[i]-startFrame[i]));
			mat[i].xHat[3] = (position[i][endFrame[i]-startFrame[i]].y()-position[i][0].y())/(g_dt*(endFrame[i]-startFrame[i]));
		}
		else 
		{
			mat[i].xHat[2] = (position[i][1].x()-position[i][0].x())/g_dt;		//velX
			mat[i].xHat[3] = (position[i][1].y()-position[i][0].y())/g_dt;		//velY
		}

		for(int s = 0; s<X_DIM; s++)
		{
			mat[i].Sigma(s,s) = initialSigma[s]; //confidence
		}

		for (size_t j = 0; j < mat[i].X.size(); ++j) {
			mat[i].X[j] = sampleGaussian(mat[i].xHat, mat[i].Sigma);
		}

		//copy to state backup
		states[i] = mat[i].xHat;
		prevStates[i] = states[i];

		//all samples
		predXd[i] = prevXd[i] = Xd[i] = mat[i].X;

		//init M;
		Matrix<X_DIM> mProcessNoise;
		for (int j=0; j<X_DIM; j++) {
			mProcessNoise[j] = processNoise[j];
		}

		M[i] = mProcessNoise * ~mProcessNoise;
		Mprev[i] = mProcessNoise * ~mProcessNoise;
		//Mprev = kf::identity<X_DIM>();
	}
}



void initData()
{
	for(int i=0; i<numAgt; i++)
	{
		position[i].clear();

		//newEM
		Xd[i].clear();
		prevXd[i].clear();
		predXd[i].clear();	
	}

}


void InitializeBRVO()
{
		initRVO();
		initKF();
		//constVelocity();
		//constAcceleration();
}

void Increment(int numAgent, int x, int y, int maxAgents)
{
	CallCount++;
	float scale = 1.01; //cm to m (Under Wildlife Testing)
	cVector2 curPos = scale*RVO::Vector2(x,y);
	position[numAgent-1].push_back(curPos);

	g_endFrame = (CallCount+1)/maxAgents;
	//g_curFrame = (CallCount+1)/maxAgents;

	if(numAgt<numAgent)
		numAgt = numAgent;

	endFrame[numAgent-1] = g_endFrame;
}

PedData RunMotionModel()
{
	return predict();
}

//PedData RunMotionModel2()
//{
//	return step();
//}