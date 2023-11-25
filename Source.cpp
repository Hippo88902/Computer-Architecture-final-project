#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
