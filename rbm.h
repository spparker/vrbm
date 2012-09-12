#ifndef RBM_H
#define RBM_H

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
using namespace std;

class Rbm
{
private:
	//visual outputs
	float* visible_out;
	float* hidden_out;

	void swap4(char* c);
	void print_img(float* img, int h, int w);
	/*void save_layer(char* output_file, float* weights, int w, int h, float* a, float* b );
	int load_layer(char* input_file, float* weights, int w, int h, float* a, float* b );
*/

public:
	//learning variables
	int epoch; //number of epochs
	int batch; //batch size
	int gibbs; //number of gibbs samples
	int steps; //number of MCMC steps
	float lr; //learning rate

	//machine variables
	int num_train;
	int num_valid;
	int v_size;	//visible layer size
	int h_size;	//hidden layer size
	int h2_size;
	int w_size;	//number of weights
	int w2_size;

	//machine parameters
	float* d_wij, *d_wjk;	//weights
	float* d_dwij, *d_dwjk;	//changes in weights
	float* d_b, *d_db, *d_a, *d_da, *d_c, *d_dc, *d_d, *d_dd; //bias' and changes in bias'
	float* d_E; //energy
	float* d_dwij_sum, *d_db_sum, *d_da_sum, *d_dc_sum, *d_dd_sum, *d_dwjk_sum; //sum of change
	//float* d_hrand;	//random number array for hidden use

	//learning layer states
	float* d_hidden0, *d_hiddenX, *d_visible0, *d_visibleX, *d_hidden20, *d_hidden2X;

	//allocation sizes
	size_t train_data_size;
	size_t convert_data_size;
	size_t wij_size, wjk_size;
	size_t hid2_size, hid_size, vis_size;

	//data
	char* dataset;
	float* train_data, *d_train_data, *d_convert_data;


	Rbm(int train, int valid, int v, int h, int h2);
	void alloc_mem();
	void init_params();
	void init_params2();
	int load_data(char* file);
	void add_convert(int i);
	
	void save_layer(char* output_file);
	void save_layer2(char* output_file);
	int load_layer(char* input_file);
	int load_layer2(char* input_file);

	void set_visible_train(int n);
	void set_visible0();
	void set_visibleX();
	float get_visible(int px);
	void set_hidden0();
	void set_hiddenX();
	float get_hidden(int px);

	void grab_sample();
	void grab_convert_sample();



	

};
#endif
