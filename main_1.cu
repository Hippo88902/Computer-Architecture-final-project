#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define pi 3.14159265359

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int BlockSize = 2;
int ThreadNum = 4;

void read_write_wav_header(FILE *input, FILE *output, int *Sample_rate, int *amount) {
	int *big = (int *)malloc(4);
	short *small = (short *)malloc(2);

	fread(big, 4, 1, input);//RIFF
	fwrite(big, 1, 4, output);

	fread(big, 4, 1, input);//ChunkSize
	fwrite(big, 1, 4, output);

	fread(big, 4, 1, input);//Wave
	fwrite(big, 1, 4, output);

	fread(big, 4, 1, input);//"fmt "
	fwrite(big, 1, 4, output);

	fread(big, 4, 1, input);//"16"
	fwrite(big, 1, 4, output);

	fread(small, 2, 1, input);//"1(PCM)"
	fwrite(small, 1, 2, output);

	fread(small, 2, 1, input);//"1(mono)"
	fwrite(small, 1, 2, output);

	fread(big, 4, 1, input);//Sample Rate(16000)
	*Sample_rate = (int)*big;
	fwrite(big, 1, 4, output);

	fread(big, 4, 1, input);//"Byte Rate(16000*1*4)"
	fwrite(big, 1, 4, output);

	fread(small, 2, 1, input);//"num of channel * byte per sample"
	fwrite(small, 1, 2, output);

	fread(small, 2, 1, input);//"bits per sample"
	fwrite(small, 1, 2, output);

	fread(big, 4, 1, input);//"data"
	fwrite(big, 1, 4, output);

	fread(big, 4, 1, input);//"num of sample * num of channel * byteper sample"
	*amount = 8 * (*big) / (int)(*small);
	fwrite(big, 1, 4, output);

	free(big); free(small);
}

void read_wav_data(FILE *input, short* input_data, int amount) {
	for (int i = 0; i < amount; i++) {
		fread((input_data + i), 1, 2, input);
	}
}

void write_wav_data(FILE *output, short* input_data, int amount) {
	for (int i = 0; i < amount; i++) {
		fwrite((input_data + i), 1, sizeof(short), output);
	}
}


struct complex {
	double real;
	double imaginary;
};

complex complex_add(complex x1, complex x2) {
	complex y;
	y.real = x1.real + x2.real; y.imaginary = x1.imaginary + x2.imaginary;
	return y;
}

complex complex_mul(complex x1, complex x2) {
	complex y;
	y.real = x1.real* x2.real - x1.imaginary*x2.imaginary; y.imaginary = x1.real* x2.imaginary + x1.imaginary*x2.real;
	return y;
}

__global__ void create_fft_table(complex fft_table[], int window_size) {
	int i = 0, TotalThread = gridDim.x*blockDim.x;
	int strip = window_size / TotalThread;
	int head = (blockIdx.x*blockDim.x + threadIdx.x)*strip;

	for (i = head; i < head + strip; i++) {
		fft_table[i].real = cos(2.0*pi*(double)i / (double)window_size);
		fft_table[i].imaginary = sin(-2.0*pi*(double)i / (double)window_size);
	}
}

__global__ void fft_stage(complex fft_input_1[], complex fft_input_2[], complex scale[], complex fft_output[],int window_size) {
	int i = 0, TotalThread = gridDim.x*blockDim.x;
	int strip = window_size / TotalThread;
	int head = (blockIdx.x*blockDim.x + threadIdx.x)*strip;

	for (i = head; i < head + strip; i++) {
		fft_output[i].real = fft_input_1[i].real + fft_input_2[i].real;
		fft_output[i].imaginary = fft_input_1[i].imaginary + fft_input_2[i].imaginary;
		fft_input_1[i].real = fft_output[i].real;
		fft_input_1[i].imaginary = fft_output[i].imaginary;

		fft_output[i].real = fft_input_1[i].real *scale[i].real - fft_input_1[i].imaginary *scale[i].imaginary;
		fft_output[i].imaginary = fft_input_1[i].real *scale[i].imaginary + fft_input_1[i].imaginary *scale[i].real;
	}
}

__global__ void create_idft_table(complex idft_table[], int window_size) {
	int i = 0, TotalThread = gridDim.x*blockDim.x;
	int strip = window_size / TotalThread;
	int head = (blockIdx.x*blockDim.x + threadIdx.x)*strip;

	for (i = head; i < head + strip; i++) {
		idft_table[i].real = cos(2.0*pi*(double)i / (double)window_size) / (double)window_size;
		idft_table[i].imaginary = sin(2.0*pi*(double)i / (double)window_size) / (double)window_size;
	}
}
__global__ void idft_stage(complex freq_signal, complex idft_scale[], complex idft_input[], complex idft_buffer[],int window_size) {
	int i = 0, TotalThread = gridDim.x*blockDim.x;
	int strip = window_size / TotalThread;
	int head = (blockIdx.x*blockDim.x + threadIdx.x)*strip;

	for (i = head; i < head + strip; i++) {
		idft_buffer[i].real = idft_scale[i].real*freq_signal.real - idft_scale[i].imaginary*freq_signal.imaginary;
		idft_buffer[i].imaginary = idft_scale[i].real*freq_signal.imaginary + idft_scale[i].imaginary*freq_signal.real;
		idft_input[i].real = idft_input[i].real + idft_buffer[i].real;
		idft_input[i].imaginary = idft_input[i].imaginary + idft_buffer[i].imaginary;
	}
}

void fft_reorder(complex *fft_window_1, complex *fft_window_2, int window_size, int stage) {
	int i, j, reverse;
	for (i = 0; i < window_size; i++) {
		reverse = 0;
		for (j = 0; j < stage; j++) {
			if (i % (int)pow(2, j + 1) >= (int)pow(2, j))
				reverse += pow(2, stage - 1 - j);
		}
		(fft_window_2 + reverse)->real = (fft_window_1 + i)->real;
		(fft_window_2 + reverse)->imaginary = (fft_window_1 + i)->imaginary;
	}
}

int main()
{
	FILE *input = fopen("Voice.wav", "rb");//存取檔案部分
	FILE *output = fopen("Output_GPU.wav", "wb");
	FILE *frequency_list = fopen("frequency.txt", "wb");

	int *amount = (int *)malloc(sizeof(int)), *Sample_rate = (int *)malloc(sizeof(int));
	short *time_signal;
	read_write_wav_header(input, output, Sample_rate, amount);
	time_signal = (short *)calloc(*amount, sizeof(short));
	read_wav_data(input, time_signal, *amount);

	int window_size, second, stage = 0;
	for (window_size = 1; window_size < *Sample_rate; window_size *= 2) stage += 1;
	for (second = 1; second*(*Sample_rate) < *amount; second++);
	dim3 dimBlock(BlockSize);
	dim3 dimGrid(ThreadNum);

	complex *fft_table = (complex *)calloc(window_size, sizeof(complex));
	complex *d_fft_table = 0; cudaMalloc((void **)&d_fft_table, sizeof(complex)*window_size);
	create_fft_table <<< dimGrid, dimBlock >>> (d_fft_table, window_size);
	cudaMemcpy(fft_table, d_fft_table, sizeof(double)*window_size, cudaMemcpyDeviceToHost);

	complex *freq_signal = (complex *)calloc(window_size*second, sizeof(complex));
	int i, j, points, s, k;
	complex *scale = (complex *)calloc(window_size, sizeof(complex));
	complex *fft_output = (complex *)calloc(window_size, sizeof(complex));
	complex *fft_input_1 = (complex *)calloc(window_size, sizeof(complex));
	complex *fft_input_2 = (complex *)calloc(window_size, sizeof(complex));

	complex *d_fft_input_1 = 0; cudaMalloc((void **)&d_fft_input_1, sizeof(complex)*window_size);
	complex *d_fft_input_2 = 0; cudaMalloc((void **)&d_fft_input_2, sizeof(complex)*window_size);
	complex *d_scale = 0; cudaMalloc((void **)&d_scale, sizeof(complex)*window_size);
	complex *d_fft_output = 0; cudaMalloc((void **)&d_fft_output, sizeof(complex)*window_size);

	for (s = 0; s < second; s++) {
		for (i = 0; i < window_size; i++) {
			if (i < *Sample_rate && *Sample_rate*s + i < *amount)
				(fft_output + i)->real = (double)*(time_signal + *Sample_rate*s + i);
			else {
				(fft_output + i)->real = 0.0;
			}
			(fft_output + i)->imaginary = 0.0;
		}

		for (i = 0; i < stage; i++) {
			k = (int)pow(2, i);
			points = (int)pow(2, stage - i);
			for (j = 0; j < window_size ; j++) {
				if ((j % points) < points / 2) {
					*(fft_input_1 + j) = *(fft_output + j);
					*(fft_input_2 + j) = *(fft_output + j + (int)points / 2);
					(scale + j)->real = 1.0; (scale + j)->imaginary = 0.0;
				}
				else {
					(fft_input_1 + j)->real = (fft_output + j)->real * -1.0;
					(fft_input_1 + j)->imaginary = (fft_output + j)->imaginary * -1.0;
					*(fft_input_2 + j) = *(fft_output + j - (int)points / 2);
					*(scale + j) = *(fft_table + ((j%points) - points / 2)*k);
				}
			}
			cudaMemcpy(d_fft_input_1, fft_input_1, sizeof(complex)*window_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_fft_input_2, fft_input_2, sizeof(complex)*window_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_scale, scale, sizeof(complex)*window_size, cudaMemcpyHostToDevice);
			fft_stage <<< dimGrid, dimBlock >>> (d_fft_input_1, d_fft_input_2, d_scale, d_fft_output, window_size);
			cudaMemcpy(fft_output, d_fft_output, sizeof(complex)*window_size, cudaMemcpyDeviceToHost);
		}
		fft_reorder(fft_output, freq_signal + s * window_size, window_size, stage);
		for (i = 0; i < window_size; i++) {
			fprintf(frequency_list, "%lf %lf j\n", (freq_signal+s*window_size + i)->real, (freq_signal+s*window_size + i)->imaginary);
		}
	}
	cudaFree(d_fft_table); cudaFree(d_fft_input_1); cudaFree(d_fft_input_2); cudaFree(d_scale); cudaFree(d_fft_output);
	free(fft_table); free(fft_input_1); free(fft_input_2); free(scale); free(fft_output); free(time_signal);

	//IDFT
	complex *idft_table = (complex *)calloc(window_size, sizeof(complex));
	complex *d_idft_table = 0; cudaMalloc((void **)&d_idft_table, sizeof(complex)*window_size);
	create_idft_table <<< dimGrid, dimBlock >>> (d_idft_table , window_size);
	cudaMemcpy(idft_table, d_idft_table, sizeof(complex)*window_size, cudaMemcpyDeviceToHost);
	cudaFree(d_idft_table);

	complex *idft_input = (complex *)calloc(window_size, sizeof(complex));
	complex *idft_scale = (complex *)calloc(window_size, sizeof(complex));
	complex *d_idft_input = 0; cudaMalloc((void **)&d_idft_input, sizeof(complex)*window_size);
	complex *d_idft_scale = 0; cudaMalloc((void **)&d_idft_scale, sizeof(complex)*window_size);
	complex *d_idft_buffer = 0; cudaMalloc((void **)&d_idft_buffer, sizeof(complex)*window_size);
	complex *idft_output = (complex *)calloc(window_size*second, sizeof(complex));
	
	for (s = 0; s < second; s++) {
		for (j = 0; j < window_size; j++) {
			(idft_input + j)->real = 0.0; (idft_input + j)->imaginary = 0.0;
		}
		cudaMemcpy(d_idft_input, idft_input, sizeof(complex)*window_size, cudaMemcpyHostToDevice);
		for (i = 0; i < window_size; i++) {
			for (j = 0; j < window_size; j++) {
				*(idft_scale + j) = *(idft_table + ((i + 1)*j) % window_size);
			}
			cudaMemcpy(d_idft_scale, idft_scale, sizeof(complex)*window_size, cudaMemcpyHostToDevice);
			idft_stage <<< dimGrid, dimBlock >>> (*(freq_signal+s*window_size+i), d_idft_scale, d_idft_input,d_idft_buffer, window_size);
			printf("%d ", i);
		}
		cudaMemcpy(idft_output+s*window_size, d_idft_input, sizeof(complex)*window_size, cudaMemcpyDeviceToHost);
		printf("%d sec done\n", s);
	}
	cudaFree(d_idft_input); cudaFree(d_idft_buffer); cudaFree(d_idft_scale);

	j = 0;
	short *restore = (short *)calloc(*amount, sizeof(short));
	for (i = 0; i < second*window_size; i++) {
		if ((i % window_size) < *Sample_rate) {
			*(restore + j) = (short)(idft_output + i)->real;
			j++;
		}
		if (j == *amount) break;
	}
	write_wav_data(output, restore, *amount);
	free(restore); free(idft_input); free(idft_table); free(idft_output);
	
	free(freq_signal);
	fclose(input); fclose(output); fclose(frequency_list);

}

