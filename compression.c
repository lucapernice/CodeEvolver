#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

// Struttura per tenere traccia delle statistiche di compressione
typedef struct {
    size_t original_size;
    size_t compressed_size;
    double compression_time;
    double decompression_time;
    size_t memory_used;
    bool integrity_check;
} CompressionStats;

// Buffer massimo per la lettura/scrittura
#define MAX_BUFFER_SIZE 1048576 // 1 MB
#define NUM_TIMING_RUNS 10 // Numero di volte per ripetere la misurazione del tempo

// Funzione di compressione da evolvere
//INIZIO_FUNZIONE_COMPRESSIONE
unsigned char* compress(const unsigned char* data, size_t data_size, size_t* compressed_size) {
   
    unsigned char* compressed = (unsigned char*)malloc(data_size * 2);
    if (compressed == NULL) return NULL;
    
    size_t compressed_index = 0;
    size_t i = 0;
    
    
    while (i < data_size) {
        unsigned char current = data[i];
        unsigned char count = 1;
        
        
        while (i + 1 < data_size && data[i + 1] == current && count < 255) {
            count++;
            i++;
        }
        
        
        compressed[compressed_index++] = count;
        compressed[compressed_index++] = current;
        
        i++;
    }
    
    
    compressed = (unsigned char*)realloc(compressed, compressed_index);
    *compressed_size = compressed_index;
    
    return compressed;
    
}
//FINE_FUNZIONE_COMPRESSIONE


// Funzione di decompressione da evolvere
//INIZIO_FUNZIONE_DECOMPRESSIONE
unsigned char* decompress(const unsigned char* compressed_data, size_t compressed_size, size_t* decompressed_size) {
    size_t max_size = compressed_size * 255; 
    unsigned char* decompressed = (unsigned char*)malloc(max_size);
    if (decompressed == NULL) return NULL;
    
    size_t decompressed_index = 0;
    size_t i = 0;
    
    // Decompressione RLE base
    while (i < compressed_size) {
        unsigned char count = compressed_data[i++];
        unsigned char value = compressed_data[i++];
        
        // Ripete il valore 'count' volte
        for (unsigned char j = 0; j < count; j++) {
            decompressed[decompressed_index++] = value;
        }
    }
    
    // Ridimensiona alla dimensione effettiva
    decompressed = (unsigned char*)realloc(decompressed, decompressed_index);
    *decompressed_size = decompressed_index;
    
    return decompressed;
}
//FINE_FUNZIONE_DECOMPRESSIONE


// Funzione per leggere dati da file
unsigned char* read_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Errore nell'apertura del file %s\n", filename);
        return NULL;
    }
    
    // Determina la dimensione del file
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocazione del buffer
    unsigned char* buffer = (unsigned char*)malloc(*size);
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }
    
    // Lettura del file
    size_t bytes_read = fread(buffer, 1, *size, file);
    fclose(file);
    
    if (bytes_read != *size) {
        free(buffer);
        return NULL;
    }
    
    return buffer;
}


// Funzione per scrivere dati su file
bool write_file(const char* filename, const unsigned char* data, size_t size) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        return false;
    }
    
    size_t bytes_written = fwrite(data, 1, size, file);
    fclose(file);
    
    return bytes_written == size;
}

// Funzione per verificare l'integrità dei dati dopo compressione/decompressione
bool verify_integrity(const unsigned char* original, size_t original_size, 
                     const unsigned char* decompressed, size_t decompressed_size) {
    if (original_size != decompressed_size) {
        return false;
    }
    
    return memcmp(original, decompressed, original_size) == 0;
}

// Funzione per eseguire e valutare compressione
CompressionStats evaluate_compression(const char* input_file) {
    CompressionStats stats = {0};
    clock_t start, end;

    // Leggi file di input
    size_t input_size;
    unsigned char* input_data = read_file(input_file, &input_size);
    if (input_data == NULL) {
        printf("Impossibile leggere il file di input\n");
        return stats; // Restituisce stats inizializzate a zero
    }

    // Copia i dati per poter usare strtok in sicurezza
    unsigned char* data_copy = (unsigned char*)malloc(input_size + 1);
    if (data_copy == NULL) {
        free(input_data);
        printf("Errore allocazione memoria per data_copy\n");
        return stats; // Restituisce stats inizializzate a zero
    }
    memcpy(data_copy, input_data, input_size);
    data_copy[input_size] = '\0'; // Assicura terminazione null

    // Variabili per calcolare le medie
    size_t total_original_size = 0;
    size_t total_compressed_size = 0;
    double total_avg_compression_time = 0; // Rinominato per chiarezza
    double total_avg_decompression_time = 0; // Rinominato per chiarezza
    size_t total_memory_used = 0;
    int word_count = 0;
    bool all_integrity_checks_passed = true;

    char* token = strtok((char*)data_copy, ","); // Primo token

    // Elabora ogni parola separatamente
    while (token != NULL) {
        size_t token_len = strlen(token);
        if (token_len == 0) { // Salta token vuoti
             token = strtok(NULL, ",");
             continue;
        }
        word_count++;
        total_original_size += token_len;

        double current_compression_time_sum = 0;
        double current_decompression_time_sum = 0;
        size_t compressed_size = 0; // Dimensione dell'ultima compressione valida
        size_t decompressed_size = 0; // Dimensione dell'ultima decompressione valida
        unsigned char* compressed_data = NULL; // Risultato dell'ultima compressione
        unsigned char* decompressed_data = NULL; // Risultato dell'ultima decompressione
        bool current_token_ok = true;

        // --- Misurazione tempo compressione ---
        for (int i = 0; i < NUM_TIMING_RUNS; ++i) {
            size_t current_run_compressed_size;
            start = clock();
            unsigned char* temp_compressed = compress((unsigned char*)token, token_len, &current_run_compressed_size);
            end = clock();

            if (temp_compressed == NULL) {
                printf("Errore durante la compressione (run %d) della parola: %s\n", i + 1, token);
                current_token_ok = false;
                break; // Esce dal ciclo di timing per questo token
            }
            current_compression_time_sum += ((double)(end - start)) / CLOCKS_PER_SEC;

            // Conserva l'ultimo risultato valido per la decompressione e l'integrità
            if (i == NUM_TIMING_RUNS - 1) {
                 compressed_data = temp_compressed; // Non liberare l'ultimo
                 compressed_size = current_run_compressed_size; // Salva la dimensione dell'ultimo
            } else {
                 free(temp_compressed); // Libera i risultati intermedi
            }
        }

        if (!current_token_ok) {
            all_integrity_checks_passed = false; // Segna fallimento generale
            token = strtok(NULL, ","); // Passa al token successivo
            continue; // Salta il resto del ciclo per questo token
        }

        // Aggiunge la dimensione dell'ultima compressione valida al totale
        total_compressed_size += compressed_size;

        // --- Misurazione tempo decompressione ---
        for (int i = 0; i < NUM_TIMING_RUNS; ++i) {
            size_t current_run_decompressed_size;
            start = clock();
            // Usa i dati compressi dall'ultima esecuzione valida della compressione
            unsigned char* temp_decompressed = decompress(compressed_data, compressed_size, &current_run_decompressed_size);
            end = clock();

            if (temp_decompressed == NULL) {
                printf("Errore durante la decompressione (run %d) della parola: %s\n", i + 1, token);
                current_token_ok = false;
                break; // Esce dal ciclo di timing per questo token
            }
            current_decompression_time_sum += ((double)(end - start)) / CLOCKS_PER_SEC;

             // Conserva l'ultimo risultato valido per l'integrità
            if (i == NUM_TIMING_RUNS - 1) {
                 decompressed_data = temp_decompressed; // Non liberare l'ultimo
                 decompressed_size = current_run_decompressed_size; // Salva la dimensione dell'ultimo
            } else {
                 free(temp_decompressed); // Libera i risultati intermedi
            }
        }

        if (!current_token_ok) {
            free(compressed_data); // Libera i dati compressi se la decompressione fallisce
            all_integrity_checks_passed = false; // Segna fallimento generale
            token = strtok(NULL, ","); // Passa al token successivo
            continue; // Salta il resto del ciclo per questo token
        }

        // Calcola tempi medi per il token corrente e aggiungi ai totali
        total_avg_compression_time += current_compression_time_sum / NUM_TIMING_RUNS;
        total_avg_decompression_time += current_decompression_time_sum / NUM_TIMING_RUNS;

        // Verifica integrità usando l'ultimo risultato della decompressione
        bool integrity_ok = verify_integrity((unsigned char*)token, token_len, decompressed_data, decompressed_size);
        if (!integrity_ok) {
            //printf("Fallimento verifica integrità per la parola: %s\n", token);
            all_integrity_checks_passed = false;
            // Non continuiamo necessariamente, ma registriamo il fallimento generale
        }

        // Calcola utilizzo memoria stimato (basato sull'ultima esecuzione valida)
        total_memory_used += compressed_size + decompressed_size;

        // Pulizia per il token corrente (libera i risultati dell'ultima run)
        free(compressed_data);
        free(decompressed_data);

        token = strtok(NULL, ","); // Prossimo token
    }

    // Calcola le statistiche finali
    if (word_count > 0) {
        stats.original_size = total_original_size;
        stats.compressed_size = total_compressed_size; // Dimensione totale compressa
        stats.compression_time = total_avg_compression_time; // Tempo medio totale
        stats.decompression_time = total_avg_decompression_time; // Tempo medio totale
        stats.memory_used = total_memory_used; // Memoria totale stimata
    } else {
         printf("Nessuna parola valida trovata nel file di input.\n");
    }

    stats.integrity_check = all_integrity_checks_passed;

    // Pulizia finale
    free(input_data);
    free(data_copy);

    return stats;
}

// Calcolo del punteggio di fitness
double calculate_fitness(CompressionStats stats) {
    if (!stats.integrity_check) {
        return 0.0; // Se l'integrità fallisce, il fitness è zero
    }
    
    // Pesi per le diverse metriche (da modificare secondo le priorità)
    const double weight_ratio = 100.0;
    const double weight_compression_time = 1.0;
    const double weight_decompression_time = 1.0;
    const double weight_memory = 10.0;
    
    // Calcola rapporto di compressione
    double compression_ratio = (double)stats.original_size / stats.compressed_size;
    
    // Formula di fitness (da adattare)
    double fitness = (weight_ratio * compression_ratio) - 
                    (weight_memory * stats.memory_used / (1024.0 * 1024.0)) - // Normalizzato in MB
                    (weight_compression_time * stats.compression_time) -
                    (weight_decompression_time * stats.decompression_time);
    return fitness;
}

// Funzione principale
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Uso: %s <file_da_comprimere>\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    char compressed_file[256];
    char decompressed_file[256];
    
    // Genera nomi per i file di output
    snprintf(compressed_file, sizeof(compressed_file), "%s.compressed", input_file);
    snprintf(decompressed_file, sizeof(decompressed_file), "%s.decompressed", input_file);
    
    // Valuta l'algoritmo di compressione
    CompressionStats stats = evaluate_compression(input_file);
    
    // Mostra risultati
    if (stats.original_size > 0) {
        printf("Dimensione originale: %zu bytes\n", stats.original_size);
        printf("Dimensione compressa: %zu bytes\n", stats.compressed_size);
        printf("Rapporto di compressione: %.6f\n", (double)stats.original_size / stats.compressed_size);
        printf("Tempo di compressione: %.6f secondi\n", stats.compression_time);
        printf("Tempo di decompressione: %.6f secondi\n", stats.decompression_time);
        printf("Controllo integrità: %s\n", stats.integrity_check ? "SUCCESSO" : "FALLITO");
        printf("Memoria utilizzata (stima): %.2f KB\n", stats.memory_used / 1024.0);
        
        printf("Punteggio fitness: %.2f\n", calculate_fitness(stats));
    }
    
    return 0;
}