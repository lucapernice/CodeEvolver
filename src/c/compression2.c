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

// Funzione di compressione da evolvere
//INIZIO_FUNZIONE_COMPRESSIONE
unsigned char* compress(const unsigned char* data, size_t data_size, size_t* compressed_size) {
    // Per i token ABC...123, possiamo usare una compressione specializzata
    // Format: [1-byte flag][1-byte len][middle_chars]
    // Flag 255 = token ABC...123, Flag 0 = fallback compression

    // Verificare se il token segue il pattern ABC...123
    if (data_size >= 6 && // ABC + 123 = minimo 6 caratteri
        data[0] == 'A' && data[1] == 'B' && data[2] == 'C' && 
        data[data_size-3] == '1' && data[data_size-2] == '2' && data[data_size-1] == '3') {
        
        // La parte centrale è quello che resta togliendo ABC e 123
        size_t middle_len = data_size - 6;  // tolgo ABC e 123
        
        // Allocare memoria per il token compresso (flag + lunghezza + caratteri centrali)
        unsigned char* compressed = (unsigned char*)malloc(2 + middle_len);
        if (compressed == NULL) return NULL;
        
        // Imposto il flag 255 per indicare la compressione specializzata
        compressed[0] = 255;
        // Memorizzo la lunghezza della parte centrale
        compressed[1] = (unsigned char)middle_len;
        
        // Copio solo la parte centrale (senza ABC...123)
        if (middle_len > 0) {
            memcpy(compressed + 2, data + 3, middle_len);
        }
        
        *compressed_size = 2 + middle_len;
        return compressed;
    }
    
    // Fallback al metodo RLE per token che non seguono il pattern
    unsigned char* compressed = (unsigned char*)malloc(data_size * 2);
    if (compressed == NULL) return NULL;
    
    // Flag 0 indica RLE standard
    compressed[0] = 0;
    
    size_t compressed_index = 1;
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
    if (compressed_size == 0) return NULL;
    
    // Controllo se è stata usata la compressione specializzata (flag 255)
    if (compressed_data[0] == 255) {
        // Recupero la lunghezza della parte centrale
        unsigned char middle_len = compressed_data[1];
        
        // Calcolo dimensione decompresso: "ABC" + middle + "123"
        size_t total_len = 3 + middle_len + 3;
        unsigned char* decompressed = (unsigned char*)malloc(total_len);
        if (decompressed == NULL) return NULL;
        
        // Inserisco il prefisso "ABC"
        decompressed[0] = 'A';
        decompressed[1] = 'B';
        decompressed[2] = 'C';
        
        // Copio la parte centrale
        if (middle_len > 0) {
            memcpy(decompressed + 3, compressed_data + 2, middle_len);
        }
        
        // Inserisco il suffisso "123"
        decompressed[3 + middle_len] = '1';
        decompressed[4 + middle_len] = '2';
        decompressed[5 + middle_len] = '3';
        
        *decompressed_size = total_len;
        return decompressed;
    }
    
    // Decompressione RLE standard
    // Ignoriamo il byte di flag (0) all'inizio
    size_t compressed_index = 1;
    
    // Stimare la dimensione massima del decompresso
    size_t max_size = (compressed_size - 1) * 255;
    unsigned char* decompressed = (unsigned char*)malloc(max_size);
    if (decompressed == NULL) return NULL;
    
    size_t decompressed_index = 0;
    
    while (compressed_index < compressed_size) {
        unsigned char count = compressed_data[compressed_index++];
        unsigned char value = compressed_data[compressed_index++];
        
        for (unsigned char j = 0; j < count; j++) {
            decompressed[decompressed_index++] = value;
        }
    }
    
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
        return stats;
    }
    
    // Copia i dati per poter inserire terminatori di stringa
    unsigned char* data_copy = (unsigned char*)malloc(input_size + 1);
    if (data_copy == NULL) {
        free(input_data);
        printf("Errore allocazione memoria\n");
        return stats;
    }
    memcpy(data_copy, input_data, input_size);
    data_copy[input_size] = '\0';
    
    // Variabili per calcolare le medie
    size_t total_original_size = 0;
    size_t total_compressed_size = 0;
    double total_compression_time = 0;
    double total_decompression_time = 0;
    size_t total_memory_used = 0;
    int word_count = 0;
    bool all_integrity_checks_passed = true;
    
    char* token = strtok((char*)data_copy, ",");
    
    // Elabora ogni parola separatamente
    while (token != NULL) {
        size_t token_len = strlen(token);
        word_count++;
        total_original_size += token_len;
        
        // Compressione
        start = clock();
        size_t compressed_size;
        unsigned char* compressed_data = compress((unsigned char*)token, token_len, &compressed_size);
        end = clock();
        
        if (compressed_data == NULL) {
            printf("Errore durante la compressione della parola: %s\n", token);
            all_integrity_checks_passed = false;
            token = strtok(NULL, ",");
            continue;
        }
        
        total_compressed_size += compressed_size;
        total_compression_time += ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Decompressione
        start = clock();
        size_t decompressed_size;
        unsigned char* decompressed_data = decompress(compressed_data, compressed_size, &decompressed_size);
        end = clock();
        
        if (decompressed_data == NULL) {
            printf("Errore durante la decompressione della parola: %s\n", token);
            free(compressed_data);
            all_integrity_checks_passed = false;
            token = strtok(NULL, ",");
            continue;
        }
        
        total_decompression_time += ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Verifica integrità
        bool integrity_ok = verify_integrity((unsigned char*)token, token_len, decompressed_data, decompressed_size);
        if (!integrity_ok) {
            
            all_integrity_checks_passed = false;
        }
        
        // Calcola utilizzo memoria stimato
        total_memory_used += compressed_size + decompressed_size;
        
        // Pulizia
        free(compressed_data);
        free(decompressed_data);
        
        token = strtok(NULL, ",");
    }
    
    // Calcola le medie
    if (word_count > 0) {
        stats.original_size = total_original_size;
        stats.compressed_size = total_compressed_size;
        stats.compression_time = total_compression_time;
        stats.decompression_time = total_decompression_time;
        stats.memory_used = total_memory_used;
    }
    
    stats.integrity_check = all_integrity_checks_passed;
    
    // Pulizia
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
    const double weight_ratio = 20.0;
    const double weight_compression_time = 1.0;
    const double weight_decompression_time = 1.0;
    const double weight_memory = 10.0;
    
    // Calcola rapporto di compressione
    double compression_ratio = (double)stats.original_size / stats.compressed_size;
    
    // Formula di fitness (da adattare)
    double fitness = (weight_ratio * compression_ratio) + 
                    (weight_memory * stats.memory_used / (1024.0 * 1024.0)) + // Normalizzato in MB
                    (weight_compression_time * stats.compression_time) +
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
        printf("Rapporto di compressione: %.2f\n", (double)stats.original_size / stats.compressed_size);
        printf("Tempo di compressione: %.6f secondi\n", stats.compression_time);
        printf("Tempo di decompressione: %.6f secondi\n", stats.decompression_time);
        printf("Controllo integrità: %s\n", stats.integrity_check ? "SUCCESSO" : "FALLITO");
        printf("Memoria utilizzata (stima): %.2f KB\n", stats.memory_used / 1024.0);
        
        printf("Punteggio fitness: %.2f\n", calculate_fitness(stats));
    }
    
    return 0;
}