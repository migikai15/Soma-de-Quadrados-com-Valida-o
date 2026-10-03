#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 40

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* N precisa ser divisível pelo número de processos */
    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Erro: N=%d nao e divisivel por %d processos.\n", N, size);
        MPI_Finalize();
        return 1;
    }

    int chunk = N / size;
    int *global = NULL;
    int *local = malloc(chunk * sizeof(int));

    /* Root cria o vetor 1..N */
    if (rank == 0) {
        global = malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            global[i] = i + 1;
    }

    /* Distribui partes iguais para todos os processos */
    MPI_Scatter(global, chunk, MPI_INT,
                local, chunk, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Exibe vetor local */
    printf("Processo %d recebeu:", rank);
    for (int i = 0; i < chunk; i++)
        printf(" %d", local[i]);
    printf("\n");
    fflush(stdout);

    /* Soma local dos quadrados */
    long long soma_local = 0;
    for (int i = 0; i < chunk; i++)
        soma_local += (long long)local[i] * local[i];

    printf("Processo %d: soma local dos quadrados = %lld\n", rank, soma_local);
    fflush(stdout);

    /* Reduz as somas locais no root */
    long long soma_paralela = 0;
    MPI_Reduce(&soma_local, &soma_paralela, 1, MPI_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Soma sequencial */
        long long soma_seq = 0;
        for (int i = 1; i <= N; i++)
            soma_seq += (long long)i * i;

        /* Fórmula fechada */
        long long formula = (long long)N * (N + 1) * (2 * N + 1) / 6;

        printf("\nProcesso 0: soma paralela dos quadrados = %lld\n", soma_paralela);
        printf("Processo 0: soma sequencial esperada    = %lld\n", soma_seq);
        printf("Processo 0: soma pela formula           = %lld\n", formula);

        if (soma_paralela == soma_seq && soma_seq == formula)
            printf("\nOs valores conferem!\n");
        else
            printf("\nOs valores NAO conferem!\n");

        free(global);
    }

    free(local);
    MPI_Finalize();
    return 0;
}