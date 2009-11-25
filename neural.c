/***************************************
**    Trabalho de Redes Neurais
**      Implementacao de MLP
**
** ICMC/USP - 2009-10-30
**
** Arthur Nascimento	5634455
** Lucas Vendramin	5961586
** Rodrigo P. Zeli	5635011
***************************************/

#define _POSIX_SOURCE

#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <error.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <alloca.h>
#include <string.h>
#include <math.h>

#include "rn.h"


/***************************************************************************************************
 Funcoes de interface e chamada continua de treinamento
***************************************************************************************************/

void ler_RN(MLP **RN, FILE *instream)
{

	float learning_rate;
	int nInput, nLayers;
	if (fscanf(instream, " %f %d %d", &learning_rate, &nInput, &nLayers) != 3)
		error(6, 0, "o arquivo da rede neural não está completo");

	int *nNeurons = xmalloc(sizeof(int) * (nLayers + 1));
	for (int i = 0; i < nLayers; i++)
		if (fscanf(instream, " %d", nNeurons + i) != 1)
			error(7, 0, "o arquivo da rede neural não está completo");
	*RN = start_vars(nLayers, nNeurons, nInput, learning_rate, sigmoid, dev_sigmoid);

}

void ler_RN_e_W(MLP **RN, FILE *instream)
{
	ler_RN(RN, instream);

	for (int k = 0; k < (*RN)->QT_LAYERS; k++)
		for (int i = 0; i < (k ? (*RN)->QT_NEU[k-1] : (*RN)->QT_INPUT) + 1; i++)
			for (int j = 0; j < (*RN)->QT_NEU[k]; j++)
				if (fscanf(instream, " %f", (*RN)->W[k][i] + j) != 1)
					(*RN)->W[k][i][j] = uniforme01 * 2.0 + 1.0;

}

void ler_treinamento(MLP *RN, FILE *instream)
{

	int qtdd;

	if (fscanf(instream, " %d", &qtdd) != 1)
		qtdd = 0;

	RN->TREINAMENTOS = xrealloc(RN->TREINAMENTOS, sizeof(float *) * (RN->QT_TREINAMENTO + qtdd));
	for (int i = RN->QT_TREINAMENTO; i < (RN->QT_TREINAMENTO + qtdd); i++)
		if ((RN->TREINAMENTOS[i] = xmalloc(sizeof(float) * (RN->QT_INPUT + RN->QT_NEU[RN->QT_LAYERS - 1]))) == NULL)
			error(35, 0, "erro ao alocar mais memória");

	for (int i = RN->QT_TREINAMENTO; i < (RN->QT_TREINAMENTO + qtdd); i++)
		for (int j = 0; j < (RN->QT_INPUT + RN->QT_NEU[RN->QT_LAYERS - 1]); j++)
			if (fscanf(instream, " %f", RN->TREINAMENTOS[i] + j) != 1)
				error(8, 0, "os dados de treinamento estão incompletos");

	RN->QT_TREINAMENTO += qtdd;
}

void mais_treinamento(MLP *RN, const char *arq)
{

	if (RN == NULL)
		error(12, 0, "a rede neural não foi inicializada");

	int fd = open(arq, O_RDONLY);

	if (fd == -1)
		error(10, errno, "erro ao abrir o arquivo de treinamento %s", arq);

	FILE *instream = fdopen(fd, "r");

	if (instream == NULL)
		error(21, errno, "erro ao chamar fdopen");

	ler_treinamento(RN, instream);

	if (fclose(instream) == EOF)
		error(47, errno, "erro ao fechar a stream do arquivo '%s'", arq);
//	if (close(fd) == -1)
//		error(48, errno, "erro ao fechar o arquivo '%s'", arq);

}

void carregar_RN(MLP **RN, const char *arq)
{

	if (*RN != NULL)
		error(2, 0, "a rede neural já foi inicializada");

	int fd = open(arq, O_RDONLY);

	if ((fd == -1) && (errno != ENOENT))
		error(5, errno, "erro ao carregar a rede neural '%s'", arq);

	if ((fd == -1) && (errno == ENOENT)) {
		ler_RN(RN, stdin);
		ler_treinamento(*RN, stdin);
	} else {
		FILE *instream = fdopen(fd, "r");

		if (instream == NULL)
			error(22, errno, "erro ao chamar fdopen");

		ler_RN_e_W(RN, instream);
		ler_treinamento(*RN, instream);

		if (fclose(instream) == EOF)
			error(45, errno, "erro ao fechar a stream do arquivo '%s'", arq);

//		if (close(fd) == -1)
//			error(46, errno, "erro ao fechar o arquivo '%s'", arq);
	}

}

void treinar(MLP *RN)
{
	float erro = 0;
	int itr = 0, progresso = 0, a;
	do {
		shuffle(RN->TREINAMENTOS, RN->QT_TREINAMENTO);

		erro = 0.0f;
		for (int i = 0; i < RN->QT_TREINAMENTO; i++)
			erro += learn(RN, RN->TREINAMENTOS[i], RN->TREINAMENTOS[i] + RN->QT_INPUT);
		erro /= (float) RN->QT_TREINAMENTO;
		itr++;
		if (verbose) {
			a = max(100.0 * erro / max_error, 100.0 * itr / max_iter);
			while (a > progresso) {
				putchar('.');
				progresso++;
			}
		}
	} while ((erro >= max_error) && (itr < max_iter));

}

void salvar_RN(MLP *RN, const char *arq)
{

	int fd = open(arq, O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);

	if (fd == -1)
		error(14, errno, "erro ao abrir o arquivo '%s' para salvamento da rede neural", arq);

	FILE *outstream = fdopen(fd, "w");

	if (outstream == NULL)
		error(15, errno, "erro ao abrir o arquivo '%s' para salvamento da rede neural", arq);

	if (fprintf(outstream, "%f\n%d\n%d\n\n", RN->lt, RN->QT_INPUT, RN->QT_LAYERS) < 0)
		error(16, 0, "erro ao escrever a rede neural no arquivo '%s'", arq);

	for (int i = 0; i < RN->QT_LAYERS; i++)
		if (fprintf(outstream, "%d ", RN->QT_NEU[i]) < 0)
			error(17, 0, "erro ao escrever as informações da rede neural no arquivo '%s'", arq);
	fputc('\n', outstream);

	for (int k = 0; k < RN->QT_LAYERS; k++) {
		for (int i = 0; i < (k ? RN->QT_NEU[k-1] : RN->QT_INPUT) + 1; i++) {
			for (int j = 0; j < RN->QT_NEU[k]; j++)
				if (fprintf(outstream, "%f ", RN->W[k][i][j]) < 0)
					error(18, 0, "erro ao escrever as matrizes de peso no arquivo '%s'", arq);
			fputc('\n', outstream);
		}
		fputc('\n', outstream);
	}

	if (fprintf(outstream, "%d\n", RN->QT_TREINAMENTO) < 0)
		error(19, 0, "erro ao escrever a quantidade de treinamentos da rede neural no arquivo '%s'", arq);

	for (int i = 0; i < RN->QT_TREINAMENTO; i++) {
		for (int j = 0; j < (RN->QT_INPUT + RN->QT_NEU[RN->QT_LAYERS - 1]); j++)
			if (fprintf(outstream, "%f ", RN->TREINAMENTOS[i][j]) < 0)
				error(20, 0, "erro ao escrever um dos treinamentos no arquivo '%s'", arq);
		fputc('\n', outstream);
	}

	if (fclose(outstream) == EOF)
		error(43, errno, "erro ao fechar a stream do arquivo '%s'", arq);
//	if (close(fd) == -1)
//		error(44, errno, "erro ao fechar o arquivo '%s'", arq);

}

void treinar_com_bmp(MLP *RN, const char *arq1, const char *arq2)
{
	if ((RN->QT_INPUT != 36) || (RN->QT_NEU[RN->QT_LAYERS - 1] != 4))
		error(32, 0, "a rede neural não tem a topologia certa para processar imagens");

	BMP_header hdr1, hdr2;
	int fd1 = open(arq1, O_RDONLY), fd2 = open(arq2, O_RDONLY);
	if ((fd1 == -1) || (fd2 == -1))
		error(26, errno, "erro ao abrir os arquivos '%s' e '%s'", arq1, arq2);

	if ((read(fd1, &hdr1, sizeof(BMP_header)) == -1) || (read(fd2, &hdr2, sizeof(BMP_header)) == -1))
		error(27, errno, "erro ao ler o cabeçalho do arquivo '%s' ou do arquivo '%s'", arq1, arq2);

	if ((hdr1.tag[0] != 'B') || (hdr1.tag[1] != 'M') || (hdr2.tag[0] != 'B') || (hdr2.tag[1] != 'M'))
		error(28, 0, "ou o arquivo '%s' (tag %c%c) ou o arquivo '%s' (tag %c%c) não é BMP", arq1, hdr1.tag[0], hdr1.tag[1], arq2, hdr2.tag[0], hdr2.tag[1]);

	if ((hdr1.bits_per_pixel != 32) || (hdr1.compression != 3) ||
			(hdr2.bits_per_pixel != 32) || (hdr2.compression != 3))
		error(29, 0, "ou o arquivo '%s' (%d bpp, %d de compressão) ou o arquivo '%s' (%d bpp, %d de compressão) não está no formato adequado", arq1, hdr1.bits_per_pixel, hdr1.compression, arq2, hdr2.bits_per_pixel, hdr2.compression);

	if ((hdr1.image_width != hdr2.image_width) || (hdr1.image_height != hdr2.image_height))
		error(30, 0, "as imagens '%s' e '%s' não são do mesmo tamanho", arq1, arq2);
	uint8_t *p;
	p = mmap(NULL, 4 * hdr1.image_width * hdr1.image_height + hdr1.data_offset, PROT_READ, MAP_SHARED, fd1, 0);
	if (p == ((void *) -1))
		error(31, errno, "erro ao mapear a imagem '%s' na memória principal", arq1);
	uint8_t *ptr1 = p + hdr1.data_offset;
	p = mmap(NULL, 4 * hdr2.image_width * hdr2.image_height + hdr2.data_offset, PROT_READ, MAP_SHARED, fd2, 0);
	if (p == ((void *) -1))
		error(36, errno, "erro ao mapear a imagem '%s' na memória principal", arq2);
	uint8_t *ptr2 = p + hdr2.data_offset;

	int tam = RN->QT_TREINAMENTO + (hdr1.image_width - 2) * (hdr1.image_height - 2);

	if ((RN->TREINAMENTOS = xrealloc(RN->TREINAMENTOS, sizeof (float *) * tam)) == NULL)
		error(33, 0, "erro ao realocar memória para o novo conjunto de treinamento");

	for (int i = RN->QT_TREINAMENTO; i < tam; i++)
		if ((RN->TREINAMENTOS[i] = xmalloc(sizeof(float) * (RN->QT_INPUT + RN->QT_NEU[RN->QT_LAYERS - 1]))) == NULL)
			error(34, 0, "erro ao alocar mais memória");

	for (unsigned int i = 1; i < (hdr1.image_width - 1); i++)
		for (unsigned int j = 1; j < (hdr1.image_height - 1); j++) {
			tam -= 1;
			int k = 0;
			for (int a = -1; a <= 1; a++)
				for (int b = -1; b <= 1; b++) {
					RN->TREINAMENTOS[tam][k] = ptr1[((j+b) * hdr1.image_width + (i+a)) * 4 + 0]/255.0f;
					RN->TREINAMENTOS[tam][k + 1] = ptr1[((j+b) * hdr1.image_width + (i+a)) * 4 + 1]/255.0f;
					RN->TREINAMENTOS[tam][k + 2] = ptr1[((j+b) * hdr1.image_width + (i+a)) * 4 + 2]/255.0f;
					RN->TREINAMENTOS[tam][k + 3] = ptr1[((j+b) * hdr1.image_width + (i+a)) * 4 + 3]/255.0f;
					k += 4;
				}
			RN->TREINAMENTOS[tam][RN->QT_INPUT] = (((uint8_t *) ptr2 +(j * hdr1.image_width + i))[0])/255.0f;
			RN->TREINAMENTOS[tam][RN->QT_INPUT+1] = (((uint8_t *) ptr2 +(j * hdr1.image_width + i))[1])/255.0f;
			RN->TREINAMENTOS[tam][RN->QT_INPUT+2] = (((uint8_t *) ptr2 +(j * hdr1.image_width + i))[2])/255.0f;
			RN->TREINAMENTOS[tam][RN->QT_INPUT+3] = (((uint8_t *) ptr2 +(j * hdr1.image_width + i))[3])/255.0f;
		}
	RN->QT_TREINAMENTO += (hdr1.image_width - 2) * (hdr1.image_height - 2);
			
	if (munmap(ptr1 - hdr1.data_offset, 4 * hdr1.image_width * hdr1.image_height + hdr1.data_offset) == -1)
		error(37, errno, "erro ao desmapear o arquivo '%s' da memória", arq1);

	if (munmap(ptr2 - hdr2.data_offset, 4 * hdr2.image_width * hdr2.image_height + hdr2.data_offset) == -1)
		error(38, errno, "erro ao desmapear o arquivo '%s' da memória", arq2);

	if (close(fd1) == -1)
		error(39, errno, "erro ao fechar o arquivo '%s'", arq1);
	if (close(fd2) == -1)
		error(40, errno, "erro ao fechar o arquivo '%s'", arq2);
}

int processar_bmp(MLP *RN, const char *arq_e)
{
	if ((RN->QT_INPUT != 36) || (RN->QT_NEU[RN->QT_LAYERS - 1] != 4))
		error(49, 0, "a rede neural não tem a topologia certa para processar imagens");

	BMP_header hdr_e, hdr_s;
	int fd_e = open(arq_e, O_RDONLY);
	if (fd_e == -1)
		error(50, errno, "erro ao abrir o arquivo da imagem '%s' a ser processada", arq_e);

	if (read(fd_e, &hdr_e, sizeof(BMP_header)) == -1)
		error(51, errno, "erro ao ler o cabeçalho do arquivo de imagem '%s'", arq_e);

	if ((hdr_e.tag[0] != 'B') || (hdr_e.tag[1] != 'M')) {
		if (close(fd_e) == -1)
			error(54, errno, "erro ao fechar o arquivo de imagem '%s'", arq_e);
		return 0;
	}

	if ((hdr_e.bits_per_pixel != 32) || (hdr_e.compression != 3))
		error(52, 0, "a imagem '%s' (%d bpp, %d de compressão) não está no formato adequado", arq_e, hdr_e.bits_per_pixel, hdr_e.compression);

	uint8_t *p;
	p = mmap(NULL, 4 * hdr_e.image_width * hdr_e.image_height + hdr_e.data_offset, PROT_READ, MAP_SHARED, fd_e, 0);
	if (p == ((void *) -1))
		error(55, errno, "erro ao mapear a imagem '%s' na memória principal", arq_e);
	uint8_t *ptr = p + hdr_e.data_offset;

	const char *prefix = "saida_";
	char *arq_s = alloca(strlen(arq_e) + sizeof(prefix));
	strcpy(arq_s, prefix);
	strcat(arq_s + sizeof(prefix), arq_e);

	int fd_s = open(arq_s, O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);
	if (fd_s == -1)
		error(58, errno, "erro ao abrir o arquivo da imagem de saída '%s'", arq_s);

	memcpy(&hdr_s, &hdr_e, sizeof(BMP_header));
	hdr_s.image_width -= 2;
	hdr_s.image_height -= 2;
	hdr_s.data_offset = sizeof(BMP_header);
	if (write(fd_s, &hdr_s, sizeof(BMP_header) != sizeof(BMP_header)))
		error(57, errno, "erro ao escrever o cabeçalho BMP da imagem de saída '%s'", arq_s);

	float entrada[RN->QT_INPUT], saida[RN->QT_NEU[RN->QT_LAYERS - 1]];
	uint8_t isaida[RN->QT_NEU[RN->QT_LAYERS - 1]];
	for (unsigned int i = 1; i < (hdr_e.image_width - 1); i++)
		for (unsigned int j = 1; j < (hdr_e.image_height - 1); j++) {
			int k = 0;
			for (int a = -1; a <= 1; a++)
				for (int b = -1; b <= 1; b++) {
					entrada[k] = ptr[((j+b) * hdr_e.image_width + (i+a)) * 4 + 0]/255.0f;
					entrada[k + 1] = ptr[((j+b) * hdr_e.image_width + (i+a)) * 4 + 1]/255.0f;
					entrada[k + 2] = ptr[((j+b) * hdr_e.image_width + (i+a)) * 4 + 2]/255.0f;
					entrada[k + 3] = ptr[((j+b) * hdr_e.image_width + (i+a)) * 4 + 3]/255.0f;
					k += 4;
				}
			execute(RN, entrada, saida);
			for (int l = 0; l < RN->QT_NEU[RN->QT_LAYERS - 1]; l++)
				isaida[l] = ((uint8_t) saida[l] * 255);

			if (write(fd_s, isaida, (sizeof(uint8_t) * RN->QT_NEU[RN->QT_LAYERS - 1])) != (ssize_t) (sizeof(uint8_t) * RN->QT_NEU[RN->QT_LAYERS - 1]))
				error(56, errno, "erro ao escrever a imagem de saída '%s'", arq_s);
		}

	if (munmap(ptr - hdr_e.data_offset, 4 * hdr_e.image_width * hdr_e.image_height + hdr_e.data_offset) == -1)
		error(53, errno, "erro ao desmapear o arquivo '%s' da memória", arq_e);

	if (close(fd_e) == -1)
		error(54, errno, "erro ao fechar o arquivo de imagem '%s'", arq_e);

	if (close(fd_s) == -1)
		error(57, errno, "erro ao fechar o arquivo de imagem de saída '%s'", arq_s);

	return 1;

}

void processar_texto(MLP *RN, const char *arq)
{

	int fd = open(arq, O_RDONLY);

	if (fd == -1)
		error(4, errno, "erro ao abrir o arquivo '%s' para processar dados", arq);

	FILE *instream = fdopen(fd, "r");

	int qtdd;

	if (fscanf(instream, " %d", &qtdd) != 1)
		qtdd = 0;

	float *input;
	float *output;
	for (int i = 0; i < qtdd; i++){
		input=(float*)xmalloc(sizeof(float)*RN->QT_INPUT);
		output=(float*)xmalloc(sizeof(float)*RN->QT_NEU[RN->QT_LAYERS - 1]);
		for (int j=0;j<RN->QT_INPUT;j++){
			if (fscanf(instream, " %f", &input[j]) != 1)
				error(9, errno, "dados para processar incorretos");
		}
		execute(RN, input, output);
		for (int j=0;j<RN->QT_NEU[RN->QT_LAYERS - 1];j++)
			printf("%f\n", output[j]);
		free(input);
		free(output);
	}

	if (fclose(instream) == EOF)
		error(41, errno, "erro ao fechar a stream do arquivo '%s'", arq);
//	if (close(fd) == -1)
//		error(42, errno, "erro ao fechar o arquivo '%s'", arq);

}

void processar_dados(MLP *RN, const char *arq)
{

	if (!processar_bmp(RN, arq))
		processar_texto(RN, arq);

}

void mostra_help(const char *prog)
{
	printf("Uso:\n\
%s RNA [-h] [-v] [-e erro_máximo] [-i máximo_iterações] [-b bmp_1 bmp_2] [-t treinamento] [arquivos_de_dados]\n\
       RNA: arquivo que irá guardar a rede neural treinada ou de onde será lida uma rede neural treinada\n\
        -h: mostra a ajuda\n\
        -v: habilita mais mensagens durante a execução\n\
        -e: define o erro máximo que precisa ser atingido\n\
        -i: define o número máximo de iterações para o treinamento\n\
        -b: usa as imagens para o treinamento\n\
            obs.: precisam estar em formato BMP 32 bits e terem as mesmas dimensões\n\
	-t: informa um arquivo que contém informações de treinamento\n\
     dados: arquivos com dados que devem ser processados pela rede neural treinada\n\
\n\
", prog);
	exit(0);

}

/***************************************************************************************************
 MAIN
***************************************************************************************************/
int main (int argc, char **argv, char **envp __attribute((unused)))
{

	MLP *myNN = NULL;
	int num_dados = 0;
	char *dados[argc];
	for (int i = 1; i < argc; i++) {
		if (argv[i][0] == '-')
			switch (argv[i][1]) {
				case 'h':
					mostra_help(argv[0]);
					break;

				case 'v':
					verbose = 1;
					break;

				case 'i':
					if (++i >= argc)
						error(24, 0, "é necessário informar o número máximo de iterações");
					if (sscanf(argv[i], "%d", &max_iter) != 1)
						error(3, 0, "informe um número para a quantidade máxima de iterações");
					break;

				case 'e':
					if (++i >= argc)
						error(11, 0, "é necessário informar o erro máximo do treinamento");
					if (sscanf(argv[i], "%f", &max_error) != 1)
						error(13, 0, "o erro foi fornecido incorretamente");
					break;

				case 'b':
					if (i+2 >= argc)
						error(23, 0, "é necessário informar duas imagens para fazer o treinamento");
					treinar_com_bmp(myNN, argv[i+1], argv[i+2]);
					i += 2;
					break;

				case 't':
					if (++i >= argc)
						error(25, 0, "é necessário informar um arquivo de treinamento");
					mais_treinamento(myNN, argv[i]);
					break;

			}
		else
			if (myNN == NULL)
				carregar_RN(&myNN, argv[1]);
			else
				dados[num_dados++] = argv[i];
	}

	if (myNN == NULL)
		error(0, 0, "é necessário carregar uma rede neural");

	treinar(myNN);

	salvar_RN(myNN, argv[1]);

	while (num_dados --> 0)
		processar_dados(myNN, dados[num_dados]);

	return 0;

}
