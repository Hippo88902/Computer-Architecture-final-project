// CA_Final.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define pi 3.14159265359

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

void create_fft_table(complex* fft_table, int window_size) {
	for (int i = 0; i < window_size; i++) {
		(fft_table + i)->real = cos(2.0*pi*(double)i / (double)window_size);
		(fft_table + i)->imaginary = sin(-2.0*pi*(double)i / (double)window_size);
	}
	printf("FFT table created!!!\n");
}

void create_idft_table(complex* idft_table, int window_size) {
	for (int j = 0; j < window_size; j++) {
		(idft_table + j)->real = cos(2.0*pi*(double)j / (double)window_size)/(double)window_size;
		(idft_table + j)->imaginary = sin(2.0*pi*(double)j / (double)window_size)/ (double)window_size;
	}
	printf("IDFT table created!!!\n");
}

void fft_butterfly(complex *fft_window_1, complex *fft_window_2, complex* fft_table, int points, int window_size) {//points =2^N
	int progress = int(window_size / points);
	for (int i = 0; i < points; i++) {
		if (i < points / 2) *(fft_window_2 + i) = complex_add(*(fft_window_1 + i), *(fft_window_1 + i + points / 2));
		else {
			(fft_window_2 + i)->real = -1.0*(fft_window_1 + i)->real;
			(fft_window_2 + i)->imaginary = -1.0*(fft_window_1 + i)->imaginary;
			*(fft_window_2 + i) = complex_add(*(fft_window_2 + i), *(fft_window_1 + i - points / 2));
			*(fft_window_2 + i) = complex_mul(*(fft_window_2 + i), *(fft_table + progress * (i - points / 2)));
		}
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

void read_write_wav_header(FILE *input, FILE *output, int *Sample_rate, int *amount);
void read_wav_data(FILE *input, short* input_data, int amount);
void write_wav_data(FILE *output, short* input_data, int amount);
double unification = pow(2, 15);

int main()
{
	FILE *input = fopen("Voice.wav", "rb");//存取檔案部分
	FILE *output = fopen("Output.wav", "wb");
	//FILE *restore_list = fopen("my_ans.txt", "wb");
	//FILE *answer_list = fopen("true.txt", "wb");
	FILE *frequency_list = fopen("frequency.txt", "wb");


	int *amount = (int *)malloc(sizeof(int)), *Sample_rate = (int *)malloc(sizeof(int));
	short *time_signal;
	read_write_wav_header(input, output, Sample_rate, amount);
	time_signal = (short *)calloc(*amount, sizeof(short));
	read_wav_data(input, time_signal, *amount);

	int window_size, second, stage = 0;
	for (window_size = 1; window_size < *Sample_rate; window_size *= 2) stage += 1;
	for (second = 1; second*(*Sample_rate) < *amount; second++);

	complex *fft_table = (complex*)calloc(window_size, sizeof(complex));
	create_fft_table(fft_table, window_size);
	complex *freq_signal = (complex*)calloc(window_size*second, sizeof(complex));

	int i, j, points, s, count;
	complex *fft_window_1 = (complex *)calloc(window_size, sizeof(complex));
	complex *fft_window_2 = (complex *)calloc(window_size, sizeof(complex));
	for (s = 0; s < second; s++) {
		for (i = 0; i < window_size; i++) {
			if (i < *Sample_rate && *Sample_rate*s + i < *amount)
				//(fft_window_1 + i)->real = (double)*(time_signal + *Sample_rate*s + i) / unification;
				(fft_window_1 + i)->real = (double)*(time_signal + *Sample_rate*s + i);
			else {
				(fft_window_1 + i)->real = 0.0;
			}
			(fft_window_1 + i)->imaginary = 0.0;
			//fprintf(answer_list, "%lf\n", (fft_window_1 + i)->real);
		}
		for (i = 0; i < stage; i++) {
			count = 0; points = pow(2, stage - i);
			while (count < window_size) {
				fft_butterfly(fft_window_1 + count, fft_window_2 + count, fft_table, points, window_size);
				count += points;
			}
			complex *temp = fft_window_1;
			fft_window_1 = fft_window_2;
			fft_window_2 = temp;//exchange
		}
		fft_reorder(fft_window_1, fft_window_2, window_size, stage);//fft_window_2
		for (i = 0; i < window_size; i++) {
			(freq_signal + i + s * window_size)->real = (fft_window_2 + i)->real;
			(freq_signal + i + s * window_size)->imaginary = (fft_window_2 + i)->imaginary;
			fprintf(frequency_list, "%lf %lf j\n", (freq_signal + i + s * window_size)->real, (freq_signal + i + s * window_size)->imaginary);
		}
	}

	free(fft_window_1); free(fft_window_2); free(time_signal); free(fft_table);
	fclose(frequency_list);
	//fclose(answer_list); fclose(input);
	
	//IDFT
	complex *idft_output = (complex*)calloc(window_size*second, sizeof(complex));
	complex *idft_table = (complex*)calloc((int)window_size, sizeof(complex));
	for (i = 0; i < window_size; i++)
		(idft_output + i)->real = 0.0;
	    (idft_output + i)->imaginary = 0.0;

	create_idft_table(idft_table, window_size);

	complex *temp = (complex*)malloc(sizeof(complex));
	for (s = 0; s < second; s++) {
		for (i = 0; i < window_size; i++) {
			for (j = 0; j < window_size; j++) {
				*temp = complex_mul(*(idft_table + ((i*j) % window_size)), *(freq_signal + s * window_size + j));
				*(idft_output + s * window_size + i) = complex_add(*temp, *(idft_output + s * window_size + i));
			}
			//fprintf(restore_list, "%lf\n", (idft_output + s * window_size + i)->real);
		}
		printf("%d sec done\n", s);
	}

	free(idft_table);
	free(freq_signal);

	j = 0;
	short *restore = (short *)calloc(*amount, sizeof(short)); 
	for (i = 0; i < second*window_size; i++) {
		if ((i % window_size) < *Sample_rate){
			*(restore + j) = (short)(idft_output+i)->real;
			j++;
		}
		if (j == *amount) break;
	}
	write_wav_data(output, restore, *amount);

	free(idft_output);
	free(restore);
	fclose(output); 
	//fclose(restore_list);
	printf("End!!!");
}