#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define blockIdx  blockIdx
#define blockDim  blockDim
#define threadIdx  threadIdx

typedef struct {
    double real;
    double imaginary;
} complex;

__global__ void fft_butterfly(complex *window_1, complex *window_2, complex *fft_table, int points, int window_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < points) {
        int k = i * (window_size / points);
        complex temp = window_1[i];
        window_1[i].real += fft_table[k].real * window_2[i].real - fft_table[k].imaginary * window_2[i].imaginary;
        window_1[i].imaginary += fft_table[k].imaginary * window_2[i].real + fft_table[k].real * window_2[i].imaginary;
        window_2[i] = temp;
    }
}

__global__ void fft_reorder(complex *window_1, complex *window_2, int window_size, int stage) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < window_size) {
        int j = 0;
        for (int k = 0; k < stage; k++) {
            j <<= 1;
            j |= (i >> k) & 1;
        }
        window_2[i] = window_1[j];
    }
}

__global__ void idft_calculation(complex *idft_output, complex *idft_table, complex *freq_signal, int window_size, int Sample_rate, int amount, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < window_size) {
        complex temp;
        idft_output[s * window_size + i].real = 0.0;
        idft_output[s * window_size + i].imaginary = 0.0;
        for (int j = 0; j < window_size; j++) {
            temp.real = freq_signal[s * window_size + j].real * idft_table[(i * j) % window_size].real - freq_signal[s * window_size + j].imaginary * idft_table[(i * j) % window_size].imaginary;
            temp.imaginary = freq_signal[s * window_size + j].real * idft_table[(i * j) % window_size].imaginary + freq_signal[s * window_size + j].imaginary * idft_table[(i * j) % window_size].real;
            idft_output[s * window_size + i].real += temp.real;
            idft_output[s * window_size + i].imaginary += temp.imaginary;
        }
    }
}

__global__ void restore_signal(short *restore, complex *idft_output, int window_size, int Sample_rate, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        if (i < Sample_rate)
            restore[i] = (short)(idft_output[i].real);
    }
}

void read_write_wav_header(FILE *input, FILE *output, int *Sample_rate, int *amount) {
    // 處理WAV檔案的header讀取和寫入
}

void read_wav_data(FILE *input, short* input_data, int amount) {
    // 讀取WAV檔案的音訊資料
}

void write_wav_data(FILE *output, short* output_data, int amount) {
    // 寫入WAV檔案的音訊資料
}

void create_fft_table(complex *fft_table, int window_size) {
    // 創建FFT表格
}

void create_idft_table(complex *idft_table, int window_size) {
    // 創建IDFT表格
}

void paralled_idct_stage(complex *freq_signal, complex *idft_output, complex *idft_table, int window_size, int i) {
    // 平行處理IDCT階段
}

int main() {
    
    FILE *input = fopen("Voice.wav", "rb");
    FILE *output = fopen("Output.wav", "wb");
    int *amount = (int *)malloc(sizeof(int));
    int *Sample_rate = (int *)malloc(sizeof(int));
    short *time_signal;
    read_write_wav_header(input, output, Sample_rate, amount);
    time_signal = (short *)calloc(*amount, sizeof(short));
    read_wav_data(input, time_signal, *amount);

    int window_size, second, stage = 0;
    for (window_size = 1; window_size < *Sample_rate; window_size *= 2)
        stage += 1;
    for (second = 1; second * (*Sample_rate) < *amount; second++);

    complex *fft_table = (complex *)calloc(window_size, sizeof(complex));
    create_fft_table(fft_table, window_size);
    complex *freq_signal = (complex *)calloc(window_size * second, sizeof(complex));

    // Allocate device memory
    complex *dev_fft_table, *dev_freq_signal;
    cudaMalloc((void **)&dev_fft_table, window_size * sizeof(complex));
    cudaMalloc((void **)&dev_freq_signal, window_size * second * sizeof(complex));

    // Copy data to device memory
    cudaMemcpy(dev_fft_table, fft_table, window_size * sizeof(complex), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (window_size + block_size - 1) / block_size;

    for (int s = 0; s < second; s++) {
        // Copy data to device memory
        complex *dev_fft_window_1, *dev_fft_window_2;
        cudaMalloc((void **)&dev_fft_window_1, window_size * sizeof(complex));
        cudaMalloc((void **)&dev_fft_window_2, window_size * sizeof(complex));
        cudaMemcpy(dev_fft_window_1, freq_signal + s * window_size, window_size * sizeof(complex), cudaMemcpyHostToDevice);

        for (int i = 0; i < stage; i++) {
            int count = 0;
            int points = pow(2, stage - i);
            while (count < window_size) {
                fft_butterfly<<<grid_size, block_size>>>(dev_fft_window_1, dev_fft_window_2, dev_fft_table, points, count);
                count += points;
            }
            // Swap the windows
            complex *temp = dev_fft_window_1;
            dev_fft_window_1 = dev_fft_window_2;
            dev_fft_window_2 = temp;
        }

        fft_reorder<<<grid_size, block_size>>>(dev_fft_window_2, dev_fft_window_1, window_size, stage);

        // Copy data back to host memory
        cudaMemcpy(freq_signal + s * window_size, dev_fft_window_2, window_size * sizeof(complex), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(dev_fft_window_1);
        cudaFree(dev_fft_window_2);
    }

    // Free device memory
    cudaFree(dev_fft_table);
    cudaFree(dev_freq_signal);

    // Perform IDFT calculations
    complex *idft_output = (complex *)calloc(window_size * second, sizeof(complex));
    complex *idft_table = (complex *)calloc(window_size, sizeof(complex));
    create_idft_table(idft_table, window_size);

    // Allocate device memory
    complex *dev_idft_output, *dev_idft_table, *dev_freq_signal;
    cudaMalloc((void **)&dev_idft_output, window_size * second * sizeof(complex));
    cudaMalloc((void **)&dev_idft_table, window_size * sizeof(complex));
    cudaMalloc((void **)&dev_freq_signal, window_size * second * sizeof(complex));

    // Copy data to device memory
    cudaMemcpy(dev_idft_table, idft_table, window_size * sizeof(complex), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_freq_signal, freq_signal, window_size * second * sizeof(complex), cudaMemcpyHostToDevice);

    for (int s = 0; s < second; s++) {
        idft_calculation<<<grid_size, block_size>>>(dev_idft_output + s * window_size, dev_idft_table, dev_freq_signal + s * window_size, window_size, *Sample_rate, *amount, s);
    }

    // Copy data back to host memory
    cudaMemcpy(idft_output, dev_idft_output, window_size * second * sizeof(complex), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_idft_output);
    cudaFree(dev_idft_table);
    cudaFree(dev_freq_signal);

    // Restore the signal
    short *restore = (short *)malloc(*amount * sizeof(short));
    cudaMemcpy(restore, dev_idft_output, *amount * sizeof(short), cudaMemcpyDeviceToHost);

    // Write the restored signal to output file
    write_wav_data(output, restore, *amount);

    // Free host memory
    free(fft_table);
    free(freq_signal);
    free(idft_output);
    free(restore);

    // Close file streams
    fclose(input);
    fclose(output);

    printf("End!!!");

    return 0;
}
