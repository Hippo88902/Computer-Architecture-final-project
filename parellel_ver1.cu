#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define pi 3.14159265359
double unification = pow(2, 15);

typedef struct {
    double real;
    double imaginary;
} complex;

__device__ complex complex_add(complex x1, complex x2) {
    complex y;
    y.real = x1.real + x2.real;
    y.imaginary = x1.imaginary + x2.imaginary;
    return y;
}

__device__ complex complex_mul(complex x1, complex x2) {
    complex y;
    y.real = x1.real * x2.real - x1.imaginary * x2.imaginary;
    y.imaginary = x1.real * x2.imaginary + x1.imaginary * x2.real;
    return y;
}

__global__ void create_fft_table(complex* fft_table, int window_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < window_size) {
        fft_table[index].real = cos(2.0 * pi * (double)index / (double)window_size);
        fft_table[index].imaginary = sin(-2.0 * pi * (double)index / (double)window_size);
    }
    printf("FFT table created!!!\n");
}

__global__ void create_idft_table(complex* idft_table, int window_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < window_size) {
        idft_table[index].real = cos(2.0 * pi * (double)index / (double)window_size) / (double)window_size;
        idft_table[index].imaginary = sin(2.0 * pi * (double)index / (double)window_size) / (double)window_size;
    }
    printf("IDFT table created!!!\n");
}

__global__ void paralled_idft_stage(complex* freq_signal, complex* idft_output, complex* idft_table, int window_size, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < window_size) {
        idft_output[idx] = complex_add(idft_output[idx], complex_mul(idft_table[(count * idx) % window_size], freq_signal[count]));
    }
}

__global__ void fft_butterfly(complex* fft_window_1, complex* fft_window_2, complex* fft_table, int points, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int progress = window_size / points;
    
    if (idx < points) {
        if (idx < points / 2) {
            fft_window_2[idx] = complex_add(fft_window_1[idx], fft_window_1[idx + points / 2]);
        } else {
            fft_window_2[idx].real = -1.0 * fft_window_1[idx].real;
            fft_window_2[idx].imaginary = -1.0 * fft_window_1[idx].imaginary;
            fft_window_2[idx] = complex_add(fft_window_2[idx], fft_window_1[idx - points / 2]);
            fft_window_2[idx] = complex_mul(fft_window_2[idx], fft_table[progress * (idx - points / 2)]);
        }
    }
}

__global__ void fft_reorder(complex* fft_window_1, complex* fft_window_2, int window_size, int stage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j, reverse;
    
    if (idx < window_size) {
        reverse = 0;
        for (j = 0; j < stage; j++) {
            if (idx % (int)powf(2, j + 1) >= (int)powf(2, j)) {
                reverse += powf(2, stage - 1 - j);
            }
        }
        fft_window_2[reverse].real = fft_window_1[idx].real;
        fft_window_2[reverse].imaginary = fft_window_1[idx].imaginary;
    }
}

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

int main() {
    // 其他程式碼...
	FILE *input = fopen("Voice.wav", "rb");//存取檔案部分
	FILE *output = fopen("Output.wav", "wb");
	//FILE *restore_list = fopen("my_ans.txt", "wb");
	//FILE *answer_list = fopen("true.txt", "wb");
	FILE *frequency_list = fopen("frequency.txt", "wb");

    // 將複數結構體移至 GPU 內存
    complex *d_fft_table, *d_freq_signal, *d_idft_output, *d_idft_table;
    cudaMalloc((void **)&d_fft_table, window_size * sizeof(complex));
    cudaMalloc((void **)&d_freq_signal, window_size * second * sizeof(complex));
    cudaMalloc((void **)&d_idft_output, window_size * second * sizeof(complex));
    cudaMalloc((void **)&d_idft_table, window_size * sizeof(complex));

    // 複製資料到 GPU
    cudaMemcpy(d_fft_table, fft_table, window_size * sizeof(complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq_signal, freq_signal, window_size * second * sizeof(complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idft_table, idft_table, window_size * sizeof(complex), cudaMemcpyHostToDevice);

    // 設定 CUDA 核心數量和執行緒數量
    int THREADS_PER_BLOCK = 16;
    int numBlocks = (window_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 gridDim(numBlocks, 1, 1);
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);

    // 執行 FFT
    for (int s = 0; s < second; s++) {
        for (int i = 0; i < window_size; i++) {
            if (i < *Sample_rate && *Sample_rate * s + i < *amount) {
                (fft_window_1 + i)->real = (double)*(time_signal + *Sample_rate * s + i);
            } else {
                (fft_window_1 + i)->real = 0.0;
            }
            (fft_window_1 + i)->imaginary = 0.0;
        }
        for (int i = 0; i < stage; i++) {
            int count = 0;
            int points = pow(2, stage - i);
            while (count < window_size) {
                fft_butterfly<<<gridDim, blockDim>>>(d_fft_window_1 + count, d_fft_window_2 + count, d_fft_table, points, window_size);
                count += points;
            }
            complex *temp = d_fft_window_1;
            d_fft_window_1 = d_fft_window_2;
            d_fft_window_2 = temp;
        }
        fft_reorder<<<gridDim, blockDim>>>(d_fft_window_1, d_fft_window_2, window_size, stage);
        cudaMemcpy(freq_signal + s * window_size, d_fft_window_2, window_size * sizeof(complex), cudaMemcpyDeviceToHost);
    }

    // 執行 IDFT
    for (int s = 0; s < second; s++) {
        for (int i = 0; i < window_size; i++) {
            paralled_idft_stage<<<gridDim, blockDim>>>(d_freq_signal, d_idft_output, d_idft_table, window_size, i, s);
        }
    }

    // 複製 IDFT 結果回 CPU
    cudaMemcpy(idft_output, d_idft_output, window_size * second * sizeof(complex), cudaMemcpyDeviceToHost);

    // 其他程式碼...

    // 釋放 GPU 內存
    cudaFree(d_fft_table);
    cudaFree(d_freq_signal);
    cudaFree(d_idft_output);
    cudaFree(d_idft_table);

    return 0;
}
