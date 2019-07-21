#include "Motion.h"

// 2. Initialization
// Input Ground truth data. It can be replaced with the sensor input.
// Generate RVO simulation with the same number of agents at the positions from the input

void initRVODemo()
{
    //set up RVO Simulator – as an initial model for EnKF
    sim = new RVO::RVOSimulator();
    sim->setTimeStep(g_dt);

    /* Specify the default parameters for agents that are subsequently added. */
    sim->setAgentDefaults(2.0f, 10, 0.01f, 10.0f, 0.2f, 10.0f);

    // add agents
    for(int i=0; i<numAgt; i++)
    {
        agentID[i] = sim->agents_.size();
        sim->addAgent(position[i][0]);
    }
}

// 3. BRVO Step.
// For each step,
// Predict:
//      f(x): result from RVO simulation, repeated for # ensembles * # Agents
// Correction:
//      Compare our prediction with new sensor input (ground truth)
//      update covariance matrix
//
// when using EM, these steps are repeated (until it converges or given number).
// Note that in step() function, EnKF steps for each agent is performed

void step()
{
    // generate new ensembles - get prediction from RVO
    // prediction for each agent
    fmeanEach_RVO();

    float diffNorm = 0;
    int cnt = 0;
    int numRepeat = 5;	// # EM Loop
    Matrix<X_DIM,X_DIM> M[numAgt];

    // EM loop: learning (correction) for each agent and refine the covariance matrix
    do {
        //EnKF for each agent
        for (int i=0; i<numAgt; i++ )
        {
            M[i] = zeros<X_DIM, X_DIM>();
            diffNorm = 0;

            //new sensor input
            mat[i].z[0] = position[i][curFrameIdx+1].x();
            mat[i].z[1] = position[i][curFrameIdx+1].y();

            std::vector<Matrix<X_DIM> > tempX = mat[i].X;
            ensembleKalmanFilter(mat[i].X, mat[i].u, mat[i].z, &f_RVO, &h);

            // compute xHat from X (ensembles), and M (covariance matrix)
            mat[i].xHat = zeros<X_DIM>();
            for (size_t j = 0; j < mat[i].X.size(); ++j) {
                mat[i].xHat += mat[i].X[j] / mat[i].X.size();

                //M – covariance matrix
                stateVector diffstates = mat[i].X[j] - predXd[i][j];
                M[i] += diffstates *~diffstates;
                diffNorm += abs(diffstates);
            }

            M[i]/=NUM_SAMPLES;

            int numPast = curFrameIdx*numRepeat + cnt;
            Mprev[i] = (M[i] + (numPast) * Mprev[i])/(numPast + 1);
            diffNorm/=NUM_SAMPLES;

            mat[i].Sigma = zeros<X_DIM,X_DIM>();
            for (size_t j = 0; j < mat[i].X.size(); ++j) {
                mat[i].Sigma += (mat[i].X[j] - mat[i].xHat)*~(mat[i].X[j] - mat[i].xHat) / mat[i].X.size();
            }

            Xd[i] = mat[i].X;
            mat[i].X = tempX;
        }
        cnt++;

    } while( cnt<numRepeat );

    //copy back states to xHat (finish EnKF for current Step)
    for(int i=0; i<numAgt; i++)
    {
        states[i]	= mat[i].xHat;
        prevXd[i] = Xd[i];
        mat[i].X = Xd[i];

        Matrix<X_DIM, X_DIM> sigma = zeros<X_DIM,X_DIM>();
        Matrix<X_DIM, X_DIM> sigma2 = zeros<X_DIM,X_DIM>();
        for (size_t j = 0; j < Xd[i].size(); ++j) {

            sigma += (mat[i].X[j] - mat[i].xHat)*~(mat[i].X[j] - mat[i].xHat) / mat[i].X.size();
            sigma2 += (predXd[i][j] - predStates[i])*~(predXd[i][j] - predStates[i])/predXd[i].size();
        }

    }

    g_step++;
    g_curFrame++;
}
