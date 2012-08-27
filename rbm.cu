#include "rbm.h"

Rbm::Rbm(int train, int valid, int v, int h)
{
	num_train =train;
	num_valid =valid;
	v_size = v;
	h_size = h;
	w_size = v*h;


	//known sizes
	wij_size = w_size * sizeof(float);
	hid_size = h_size * sizeof(float);
	vis_size = v_size * sizeof(float);

	//constants for now
	epoch = 40; //50
	batch = 10;
	gibbs = 10;
	steps = 5;
	lr = 0.0005;

}

void Rbm::alloc_mem()
{
	//weights
	cudaMalloc((void**)&d_wij,wij_size);
	cudaMalloc((void**)&d_dwij,wij_size);

	//bias
	cudaMalloc((void**)&d_a,vis_size);
	cudaMalloc((void**)&d_b,hid_size);
	cudaMalloc((void**)&d_da,vis_size);
	cudaMalloc((void**)&d_db,hid_size);

	//Setup Visible & Hidden Units
	cudaMalloc((void**)&d_hidden0,hid_size);
	cudaMalloc((void**)&d_hiddenX,hid_size);
	cudaMalloc((void**)&d_visibleX,vis_size);

	//Energy and convergence checks
	cudaMalloc((void**)&d_E,sizeof(float));
	cudaMalloc((void**)&d_dwij_sum, sizeof(float));
	cudaMalloc((void**)&d_da_sum, sizeof(float));
	cudaMalloc((void**)&d_db_sum, sizeof(float));

	//Setup host memory for displaying
	hidden_out = (float*) malloc(hid_size);
	visible_out = (float*) malloc(vis_size);

}

void Rbm::init_params()
{

	//Setup Weights | i->j | Visible 0 -> Hidden 1
	float* wij = (float*)malloc(wij_size);
	srand((unsigned)time(0));
	//Approximate Gaussian u=0 z=0.01
	for (int i=0;i<w_size;i++)
	{
		float central = 0;
		for(int c=0;c<100;c++)
		{
			float u = (float)rand() / (float)RAND_MAX;
			central+= (2*u -1)*0.1;
		}
		central /= 100;
		wij[i] = central;
	}

	cudaMemcpy(d_wij, wij, wij_size,cudaMemcpyHostToDevice);
	//free(wij);

	//Hidden Bias = -2 to encourage sparsity
	float bj[h_size];
	for (int j=0;j<h_size;j++)
	{
		bj[j] = -2.0;
	}

	//Visible Bias = log[Pi/(1 - Pi)]
	float ai[v_size];
	for (int i=0;i<v_size;i++)
	{
		int sum = 0;
		//Calc Pi
		for (int n=0;n<num_train;n++)
		{
			sum += train_data[n*v_size + i];
		}
		//avoid log(0) = -inf
		if(sum == 0)
			sum = 1;
		float Pi = (float)sum / num_train;
		ai[i] = log(Pi / (1-Pi));
		//printf("Pi=%f | ai[%d])=%f\n",Pi,i,log(Pi / (1-Pi)));
	}

	//Setup Bias
	cudaMemcpy(d_a, ai,vis_size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, bj,hid_size,cudaMemcpyHostToDevice);

	//float *d_hrand;
	//cudaMalloc((void **)&d_hrand, hid_size);

	return;
	//Clean up
	//free(ai);
	//free(bj);
}

void Rbm::save_layer(char* output_file)
{
	printf("Saving Layer...");
	ofstream o_file;
	o_file.open(output_file, ios::binary);

	//place data on host
	float* wij = (float*) malloc(wij_size);
	float bj[h_size];
	float ai[v_size];

	cudaMemcpy(wij, d_wij, wij_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(ai, d_a,vis_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(bj, d_b,hid_size,cudaMemcpyDeviceToHost);

	if(o_file.is_open())
	{
		o_file.write((char*)wij,wij_size);
		o_file.seekp(wij_size);
		o_file.write((char*)ai,vis_size);
		o_file.seekp(wij_size + vis_size);
		o_file.write((char*)bj,hid_size);
		o_file.close();
		printf("Completed\n");
	}
	else
		printf("Failed\n");
	return;
}

int Rbm::load_layer(char* input_file)
{
	printf("Loading Layer from %s...",input_file);
	ifstream i_file;
	i_file.open(input_file, ios::binary);
	if(i_file.is_open())
	{
		float* wij = (float*) malloc(wij_size);
		float bj[h_size];
		float ai[v_size];
		//load weights
		for(int n=0;n<w_size;n++)
		{
			i_file.seekg(n*sizeof(float));
			i_file.read((char*)&wij[n],sizeof(float));
			//printf("w[%d]=%f ",n,w[n]);
		}
		//load a
		for(int i=0;i<v_size;i++)
		{
			i_file.seekg(sizeof(float)*h_size*v_size + i*sizeof(float));
			i_file.read((char*)&ai[i],sizeof(float));
			//printf("a[%d]=%f\t",i,a[i]);
		}
		printf("\n");
		//load b
		for(int j=0;j<h_size;j++)
		{
			i_file.seekg(sizeof(float)*h_size*v_size + v_size*sizeof(float) + j*sizeof(float));
			i_file.read((char*)&bj[j],sizeof(float));
			//printf("b[%d]=%f\t",j,b[j]);
		}
		i_file.close();

		cudaMemcpy(d_wij, wij, wij_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_a, ai,vis_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, bj,hid_size,cudaMemcpyHostToDevice);


		printf("Completed!\n");
		return 1;
	}
	else
	{
		printf("Failed to open file\n");
		return 0;
	}
}

//Grabs a random sample and sets as visible layer
void Rbm::grab_sample()
{
	int sample = rand() % (num_train - num_valid);
	d_visible0 = &d_train_data[sample*v_size];

	return;
}

//Sets visible output to a training data item
void Rbm::set_visible_train(int n)
{
	d_visible0 = &d_train_data[n*v_size];
	set_visible0();
}

void Rbm::set_visible0()
{
	cudaMemcpy(visible_out,d_visible0,vis_size, cudaMemcpyDeviceToHost);
	//print_img(visible_out, 28, 28);

}

void Rbm::set_visibleX()
{
	cudaMemcpy(visible_out,d_visibleX,vis_size, cudaMemcpyDeviceToHost);
	//print_img(visible_out, 28, 28);

}

//grabs a pixel from the visible output
float Rbm::get_visible(int px)
{
	if(px<v_size)
		return visible_out[px];

	return 0;
}

//Copies the H0 data to output hidden
void Rbm::set_hidden0()
{
	cudaMemcpy(hidden_out,d_hidden0,hid_size,cudaMemcpyDeviceToHost);
	//print_img(hidden_out, 16, 32);
}

void Rbm::set_hiddenX()
{
	cudaMemcpy(hidden_out,d_hiddenX,hid_size,cudaMemcpyDeviceToHost);
	//print_img(hidden_out, 16, 32);
}

//grabs a pixel from the hidden output
float Rbm::get_hidden(int px)
{
	if(px<h_size)
		return hidden_out[px];

	return 0;
}

/*------------------------------------------------------------------
 *  							LOAD DATA
 * Loads input data from file
 * input_file | char* | name of file containing data
 * data | int* | memory location to store data
 ------------------------------------------------------------------*/
int Rbm::load_data(char* file)
{

	ifstream i_img_train;
	i_img_train.open(file, ios::binary);

	if(i_img_train.is_open())
	{
		printf("Loading Images...\n");
		//MAGIC NUMBER
		char c_magic_number[4];
		int i_magic_number;
		i_img_train.read(c_magic_number,4);
		swap4(c_magic_number);
		memcpy(&i_magic_number,c_magic_number,4);
		printf("magic number: %d\n",i_magic_number);
		//# IMAGES
		char c_images[4];
		int i_images;
		i_img_train.seekg(4);
		i_img_train.read(c_images,4);
		swap4(c_images);
		memcpy(&i_images,c_images,4);
		printf("images: %d\n",i_images);
		//ROWS
		char c_rows[4];
		int i_rows;
		i_img_train.seekg(8);
		i_img_train.read(c_rows,4);
		swap4(c_rows);
		memcpy(&i_rows,c_rows,4);
		printf("rows: %d\n",i_rows);
		//COLUMNS
		char c_cols[4];
		int i_cols;
		i_img_train.seekg(12);
		i_img_train.read(c_cols,4);
		swap4(c_cols);
		memcpy(&i_cols,c_cols,4);
		printf("columns: %d\n",i_cols);
	}
	else
	{
		printf("Couldn't open file. Exiting...");
		return 0;
	}
	//Grab Images
	int input_ptr = 16;
	char* c_img;
	c_img = (char*)malloc(v_size*sizeof(char));

	//Allocate Training Data Space
	train_data_size = num_train * v_size * sizeof(float);
	train_data = (float*)malloc(train_data_size);

	//printf("Allocation fine");

	//Image #
	for(int n=0;n<num_train;n++)
	{
		i_img_train.seekg(input_ptr);
		i_img_train.read(c_img,v_size);
		input_ptr+=v_size;
		//Down Column
		//printf("read in");
		for(int i=0;i<v_size;i++)
		{
			//convert to 'binary'
			if (c_img[i] != 0x00)
				c_img[i] = 0x01;

			//memcpy(&data[ n*V_SIZE + i*28 + j ],&c_img[i*28 + j],1);
			train_data[ n*v_size + i] = (float)c_img[i];
		}
		//printf("copied");
		//TEST PRINT
		//print_img(&data[n*V_SIZE], 28, 28);
	}

	i_img_train.close();

	//Copy to Device
	cudaMalloc((void**)&d_train_data,train_data_size);
	cudaMemcpy(d_train_data, train_data,train_data_size,cudaMemcpyHostToDevice);

	//free mem
	free(c_img);

	printf("Data Load Complete!\n");
	return 1;
}

/*------------------------------------------------------------------
 * 							PRINT IMAGE
 * Prints a 2D Image
 * data | int* | data array in memory
 * n | int | item number in array to print
 * h | int | height of image
 * w | int | width of image
 ------------------------------------------------------------------*/
void Rbm::print_img(float* data, int h, int w)
{
	//Down Column
	for(int i=0;i<h;i++)
	{
		//Across Row
		for(int j=0;j<w;j++)
		{
			if(data[i*w + j]<=0.2)
				printf("..");
			else if(data[i*w + j]<=0.4)
				printf("++");
			else if(data[i*w + j]<=0.6)
				printf("00");
			else if(data[i*w + j]<=0.8)
				printf("&&");
			else
				printf("##");
		}
		printf("\n");
	}
	printf("\n\n");

}

/* ---------------------------------------------------------------
 *		SWAP 4
 * Non-Intel -> Intel Byte formatting for 4 bytes
 * c | char* | pointer to 4 char array
 ------------------------------------------------------------------*/

void Rbm::swap4(char* c)
{
	char tmp[4];
	tmp[0] = c[3];
	tmp[1] = c[2];
	c[3] = c[0];
	c[2] = c[1];
	c[1] = tmp[1];
	c[0] = tmp[0];
	return;
}


