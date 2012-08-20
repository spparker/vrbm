/* Visual Restricted Boltzman Machine Test
**
** Sean Parker
** 7/10/2012
*/
#include <list>

#include "matrix.h"
#include "rbm.h"

#include "GL/freeglut.h"
#include "GL/gl.h"

#define MAX_HIST 100

//CUDA Functions
__global__ void calcHb(float* v0, float* h1, float* bj, float* wij, float* rnd, int h, int w);
__global__ void calcVp(float* h1, float* v0, float* ai, float* wij, int h, int w);
__global__ void calcHp(float* v0, float* h1, float* bj, float* wij, int h, int w);

__global__ void zeroDw(float* dwij, int h, int w);
__global__ void calcDw(float* v0, float* h0, float* v1, float* h1, float* dwij, int h, int w);
__global__ void updateW(float* dwij, float* W, float rate, int batch, int h, int w);

__global__ void zeroDbias(float* b);
__global__ void calcDbias(float* v0, float* v1, float* Dbias);
__global__ void updateBias(float* Dbias, float* b, float rate, int batch);


__global__ void calcE(float* vis, float* hid,  float* E, int h, int w);
__global__ void sumDelta(float* delta, float* sum, int num);


//Drawing Functions
void show_hidden();
void show_visible();
void show_weight(int num, int *w);
void show_converge(float a, float b, float w);
void histo_w(float w);

Rbm* my_rbm;
curandGenerator_t d_rand;
float* d_hrand;

list<float> dw_hist;
int n;
int e;

void renderFunction()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    //Single EPOCH
    //for(int n=0;n<(my_rbm->num_train/my_rbm->batch);n++)
    {
		//Zero deltas
		zeroDw<<<16,32>>>(my_rbm->d_dwij, my_rbm->v_size, my_rbm->h_size);
		zeroDbias<<<1,my_rbm->v_size>>>(my_rbm->d_da);
		zeroDbias<<<1,my_rbm->h_size>>>(my_rbm->d_db);

		for(int b=0;b<my_rbm->batch;b++)
		{
			my_rbm->grab_sample();
			//show_visible();

		   curandGenerateUniform(d_rand, (float *) d_hrand, my_rbm->h_size);
		   calcHb<<<16,32>>>(my_rbm->d_visible0, my_rbm->d_hidden0, my_rbm->d_b, my_rbm->d_wij, d_hrand, my_rbm->v_size, my_rbm->h_size); //H0 (MCMC Step with sampled binary units.)
		   //my_rbm->set_hidden0();
		   //show_hidden();

			calcVp<<<8,98>>>(my_rbm->d_hidden0, my_rbm->d_visibleX, my_rbm->d_a, my_rbm->d_wij, my_rbm->v_size, my_rbm->h_size);
			//my_rbm->set_visibleX();
			//show_visible();

			calcHp<<<16,32>>>(my_rbm->d_visibleX, my_rbm->d_hiddenX, my_rbm->d_b, my_rbm->d_wij, my_rbm->v_size, my_rbm->h_size); //HX (MCMC Step with P(Vx|Ho)-> P(Hx|Vx)
			//my_rbm->set_hiddenX();
			//show_hidden();

			calcDw<<<16,32>>>(my_rbm->d_visible0, my_rbm->d_hidden0, my_rbm->d_visibleX, my_rbm->d_hiddenX, my_rbm->d_dwij, my_rbm->v_size, my_rbm->h_size); //Change in weights based on VoHo - VxHx
			calcDbias<<<1,my_rbm->v_size>>>(my_rbm->d_visible0, my_rbm->d_visibleX, my_rbm->d_da); //Change in bias TO visible units Vo - Vx
			calcDbias<<<1,my_rbm->h_size>>>(my_rbm->d_hidden0, my_rbm->d_hiddenX, my_rbm->d_db); //Chane in bias TO hidden units Ho - Hx

		}

		//Calculate Updates
		updateW<<<16,32>>>(my_rbm->d_dwij, my_rbm->d_wij, my_rbm->lr, my_rbm->batch, my_rbm->v_size, my_rbm->h_size);
		updateBias<<<1,my_rbm->v_size>>>(my_rbm->d_da, my_rbm->d_a, my_rbm->lr, my_rbm->batch);
		updateBias<<<1,my_rbm->h_size>>>(my_rbm->d_db, my_rbm->d_b, my_rbm->lr, my_rbm->batch);

		sumDelta<<<1,1>>>(my_rbm->d_dwij, my_rbm->d_dwij_sum, my_rbm->w_size);
		sumDelta<<<1,1>>>(my_rbm->d_da, my_rbm->d_da_sum, my_rbm->v_size);
		sumDelta<<<1,1>>>(my_rbm->d_db, my_rbm->d_db_sum, my_rbm->h_size);

		float sum_a, sum_b, sum_w;

		cudaMemcpy(&sum_a, my_rbm->d_da_sum, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&sum_b, my_rbm->d_db_sum, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&sum_w, my_rbm->d_dwij_sum, sizeof(float), cudaMemcpyDeviceToHost);

		//show_converge(sum_a, sum_b, sum_w);

		//histo_w(sum_w);

		//Test 1 step on sample
		my_rbm->set_visible_train(10);
		calcHb<<<16,32>>>(my_rbm->d_visible0, my_rbm->d_hidden0, my_rbm->d_b, my_rbm->d_wij, d_hrand, my_rbm->v_size, my_rbm->h_size); //H0 (MCMC Step with sampled binary units.)
		calcVp<<<8,98>>>(my_rbm->d_hidden0, my_rbm->d_visibleX, my_rbm->d_a, my_rbm->d_wij, my_rbm->v_size, my_rbm->h_size);
		my_rbm->set_visibleX();
		show_visible();




		//Calculate energy of validation set
		/*float zero = 0;
		cudaMemcpy(d_E, zero,sizeof(float),cudaMemcpyHostToDevice); //zero energy
		for(int v=1;v<=my_rbm->num_valid;v++)
		{
			my_rbm->set_visible_train(num_train-v);
			calcHb<<<16,32>>>(my_rbm->d_visible0, my_rbm->d_hidden0, my_rbm->d_b, my_rbm->d_wij, d_hrand, my_rbm->v_size, my_rbm->h_size); //H0 (MCMC Step with sampled binary units.)
			calcE<<<1,1>>>(my_rbm->d_visible0, my_rbm->d_hidden0, my_rbm->d_E, my_rbm->v_size, my_rbm->h_size);


		}*/

		   /*float* tmp = (float*)malloc(my_rbm->hid_size);
		   cudaMemcpy(tmp, my_rbm->d_hidden0,my_rbm->hid_size,cudaMemcpyDeviceToHost);*/
		//show weights
		/*int test[30] = {1200,50,450,3884,23,4394,666,8888, 9430, 3430,
						56054, 45423, 42524, 42352, 43267, 94543, 43530, 23454, 4958, 90394,
						43, 84732, 12395, 34519, 23414, 68759, 7948, 398, 9548, 20934};
		show_weight(30,test);*/

		glFlush();
    }

    //Epoch
    n++;
    if(n < (my_rbm->num_train/my_rbm->batch))
    {
    	//printf("n=%d",n);
    	glutPostRedisplay();
    }
    else
    {
    	printf("Epoch %d complete!\n",e);
    	//new epoch
    	n=0;
    	e++;

    	//adjust learning parameters
    	my_rbm->lr -= 0.001;

    	if(e < my_rbm->epoch)
    		glutPostRedisplay();
    	else
    		my_rbm->save_layer("rbm_2.param");
    }
    //glutPostRedisplay();
}

/* Main method - main entry point of application
the freeglut library does the window creation work for us,
regardless of the platform. */
int main(int argc, char** argv)
{
	//-------------------------------------
	// CUDA Setup
	//-------------------------------------
	my_rbm = new Rbm(60000,6000,784,512);
	my_rbm->alloc_mem();
	my_rbm->load_data("train-images.idx3-ubyte");
	my_rbm->init_params();
	//if(!my_rbm->load_layer("rbm_1.param"))
		//return -1;

	curandCreateGenerator(&d_rand, CURAND_RNG_PSEUDO_MTGP32);
	srand((unsigned)time(0));
	int seed = (rand() % 1000);
	curandSetPseudoRandomGeneratorSeed(d_rand, seed);
    cudaMalloc((void **)&d_hrand, my_rbm->hid_size);




	//my_rbm->set_visible_train(1);
	//-------------------------------------
	// GLUT Setup
	//-------------------------------------
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(500,500);
    glutInitWindowPosition(100,100);
    glutCreateWindow("Visual Restricted Boltzmann Machine");
    glutDisplayFunc(renderFunction);

    e=0;
    n=0;
    glutMainLoop();



      /* float* tmp = (float*)malloc(my_rbm->hid_size);
       cudaMemcpy(tmp, my_rbm->d_hidden0,my_rbm->hid_size,cudaMemcpyHostToDevice); */


    return 0;
}


/*===================================================================
 * 			DRAWING FUNCTIONS
 ===================================================================*/

void histo_w(float w)
{
	//printf("dw = %f ", w);
	dw_hist.push_back(w/100000 - 1.0f);

	if(dw_hist.size() > MAX_HIST)
		dw_hist.pop_front();

	glColor3f(0.0,0.0,0.0);
	glPointSize(10.0f);
	glEnable(GL_POINT_SMOOTH);

/*	glBegin(GL_POINTS);
		list<float>::iterator i;
		float count = 0;
		for(i=dw_hist.begin(); i != dw_hist.end(); i++)
		{
			float x = ( ( count * ( 2.0f / (float)MAX_HIST) ) - 1.0f);
			printf("x=%f\t",x);
			glVertex2f(x,(*i));
			count+=1.0;
		}
	glEnd();*/

	glBegin(GL_LINES);
			list<float>::iterator i;
			float count = 0;
			float prev_w = *(dw_hist.begin());
			float prev_x = -1.0;
			for(i=dw_hist.begin(); i != dw_hist.end(); i++)
			{
				float x = ( ( count * ( 2.0f / (float)MAX_HIST) ) - 1.0f);
				glVertex2f(prev_x, prev_w);
				glVertex2f(x,(*i));

				prev_w = (*i);
				prev_x = x;

				count+=1.0;
			}
		glEnd();

	return;

}

void show_hidden()
{
	float gridX = 32;
	    float gridY = 16;
	    float px_size = 2.0/gridX;
		for(int i=0;i<gridY;i++)
		{
			for(int j=0;j<gridX;j++)
			{
				glColor3f(my_rbm->get_hidden(i*gridX + j),my_rbm->get_hidden(i*gridX + j),my_rbm->get_hidden(i*gridX + j));
				float v_off = 1.0-(float)(i+1)*px_size;
				float h_off = -1.0+(float)j*px_size;
	  	 		glBegin(GL_POLYGON);
	        			glVertex2f(h_off, v_off);
	        			glVertex2f(h_off, v_off + px_size);
	        			glVertex2f(h_off + px_size, v_off + px_size);
	        			glVertex2f(h_off + px_size, v_off);
	    			glEnd();
			}
		}

}

void show_visible()
{

	  float grid = 28;
	    float px_size = 2.0/grid;
		for(int i=0;i<grid;i++)
		{
			for(int j=0;j<grid;j++)
			{
				glColor3f(my_rbm->get_visible(i*28 + j),my_rbm->get_visible(i*28 + j),my_rbm->get_visible(i*28 + j));
				float v_off = 1.0-(float)(i+1)*px_size;
				float h_off = -1.0+(float)j*px_size;
	  	 		glBegin(GL_POLYGON);
	        			glVertex2f(h_off, v_off);
	        			glVertex2f(h_off, v_off + px_size);
	        			glVertex2f(h_off + px_size, v_off + px_size);
	        			glVertex2f(h_off + px_size, v_off);
	    			glEnd();
			}
		}

}

void show_weight(int num, int *w)
{

	//Setup graph
	glColor3f(0.0, 0.0, 0.0);
	glBegin(GL_LINES);
		glVertex2f(-1.0, 0.0);
		glVertex2f(1.0, 0.0);
	glEnd();


	float w_val;
	float buffer = 0.01;
	float width = (2.0/(float)num) - buffer;

	for(int i=0;i<num;i++)
	{
		//grab weights
		cudaMemcpy(&w_val, &my_rbm->d_wij[w[i]], sizeof(float), cudaMemcpyDeviceToHost);

		float h_off = -1.0 + i*(width+buffer);
		//display weights
		glBegin(GL_POLYGON);
			glVertex2f(h_off, 0);
			glVertex2f(h_off, w_val * 10);
			glVertex2f(h_off + width, w_val * 10);
			glVertex2f(h_off + width, 0);
		glEnd();
	}

	return;
}

void show_converge(float a, float b, float w)
{
	//Setup graph
	glColor3f(0.0, 0.0, 0.0);
	glBegin(GL_LINES);
		glVertex2f(-1.0, 0.0);
		glVertex2f(1.0, 0.0);
	glEnd();

	//display a
	glBegin(GL_POLYGON);
		glVertex2f(-0.9, 0);
		glVertex2f(-0.9, a / 100);
		glVertex2f(-0.4, a / 100);
		glVertex2f(-0.4, 0);
	glEnd();

	//display b
	glBegin(GL_POLYGON);
		glVertex2f(-0.25, 0);
		glVertex2f(-0.25, b / 100);
		glVertex2f(0.25, b / 100);
		glVertex2f(0.25, 0);
	glEnd();

	//display w
	glBegin(GL_POLYGON);
		glVertex2f(0.4, 0);
		glVertex2f(0.4, w / 10000);
		glVertex2f(0.9, w / 10000);
		glVertex2f(0.9, 0);
	glEnd();

	return;
}

/*===================================================================
 * 			CUDA FUNCTIONS
 ===================================================================*/


/*------------------------------------------------------------------
 *  							CALC Hb
 * Stochastic determination of state of a single hidden unit given v0
 * v0 | int* | visible vector
 * h1 | int* | hidden vector
 * wij | float* | weights connection hidden and visible
 ------------------------------------------------------------------*/
__global__ void calcHb(float* v0, float* h1, float* bj, float* wij, float* rnd, int h, int w)
{
	int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float sum = bj[h_idx];
	for(int i=0;i<h;i++)
	{
		sum += v0[i] * wij[ i*w + h_idx];
	}
	float prob = 1 / (1 + __expf(-1 * sum));

	//printf("p(H[%d]=1|v) = %f <> %f\n",h_idx, prob, rnd[h_idx]);

	h1[h_idx] = (prob > rnd[h_idx]);

}

/*------------------------------------------------------------------
 *  							CALC Vp
 * Probability of a single hidden unit given h0
 * v0 | float* | visible vector
 * h1 | float* | hidden vector
 * wij | float* | weights connection hidden and visible
 ------------------------------------------------------------------*/
__global__ void calcVp(float* h1, float* v0, float* ai, float* wij, int h, int w)
{
	int v_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float sum = ai[v_idx];
	for(int j=0;j<w;j++)
	{
		sum += h1[j] * wij[ v_idx*w + j];
	}
	v0[v_idx] = 1 / (1 + __expf(-1 * sum));
	//printf("p(V0[%d]=1|h) = %f\n",v_idx,v0[v_idx]);
}

/*------------------------------------------------------------------
 *  							CALC Hp
 * Probability of a single hidden unit given v0
 * v0 | float* | visible vector
 * h1 | float* | hidden vector
 * wij | float* | weights connection hidden and visible
 ------------------------------------------------------------------*/
__global__ void calcHp(float* v0, float* h1, float* bj, float* wij, int h, int w)
{
	int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float sum = bj[h_idx];
	for(int i=0;i<h;i++)
	{
		sum += v0[i] * wij[ i*w + h_idx];
	}
	h1[h_idx] = 1 / (1 + __expf(-1 * sum));
	//printf("p(H1[%d]=1|v) = %f\n",h_idx,h1[h_idx]);
}


/*------------------------------------------------------------------
 *  							CALC DW
 * determines neg-pos difference for weight update
 * Each thread updates an entire column
 *
 ------------------------------------------------------------------*/
__global__ void calcDw(float* v0, float* h0, float* v1, float* h1, float* dwij, int h, int w)
{
    int j = threadIdx.x + blockIdx.x*blockDim.x;
    for(int i=0;i<h;i++)
    {
            dwij[i*w+j] += (v0[i]*h0[j]) - (v1[i]*h1[j]);
            //printf("(%f)", dwij[i*H1_SIZE+j]);
            /*if(j == 50 && i>200 && i<205)
                printf("dw[%d,%d]=%f\t",i,j,dwij[i*h+j]);
            if(j== 50 && i==205)
            	printf("dw[%d,%d]=%f\n",i,j,dwij[i*h+j]);*/
    }
    return;
}
/********************************
        Helper to Zero out dW
*********************************/
__global__ void zeroDw(float* dwij, int h, int w)
{
        int j = threadIdx.x + blockIdx.x*blockDim.x;
        for(int i=0;i<h;i++)
        {
                dwij[i*w+j] = 0;
        }
        return;
}
/*------------------------------------------------------------------
 *  							UPDATE W
 * Adds calculated changes to weights
 * Each thread updates an entire column
 *
 ------------------------------------------------------------------*/
__global__ void updateW(float* dwij, float* W, float rate, int batch, int h, int w)
{
    int j = threadIdx.x + blockIdx.x*blockDim.x;
    for(int i=0;i<h;i++)
    {
            W[i*w+j] += rate * (dwij[i*w+j]/ (float)batch);

        /*if(j == 50 && i>200 && i<205)
            printf("w[%d,%d]=%f\t",i,j,W[i*w+j]);
        if(j== 50 && i==205)
        	printf("w[%d,%d]=%f\n",i,j,W[i*w+j]);*/

    }
    return;
}



__global__ void calcDbias(float* v0, float* v1, float* Dbias)
{
	Dbias[threadIdx.x] += v0[threadIdx.x] - v1[threadIdx.x];
	//printf("bias[%d]=%f\n",threadIdx.x,Dbias[threadIdx.x]);
}
__global__ void updateBias(float* Dbias, float* b, float rate, int batch)
{
	b[threadIdx.x] += (Dbias[threadIdx.x]/ batch) * rate;
}
__global__ void zeroDbias(float* Dbias)
{
	Dbias[threadIdx.x] = 0;
}

//Calc Energy of the system
__global__ void calcE(float* vis, float* hid,  float* E, int h, int w)
{
	return;
}
__global__ void sumDelta(float* delta, float* sum, int num)
{
	sum[0] = 0;
	for (int i=0;i<num;i++)
		sum[0]+=abs(delta[i]);
	//printf("Sum of Delta: %f\t", sum[0]);
	//if(num>1000)
		//printf("\n");

	return;

}

