#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#define PREFIX "ABC"
#define SUFFIX "123"
#define MAX_MIDDLE_LENGTH 4
#define MAX_STRING_LENGTH 100

// Function to generate a random alphanumeric character
char getRandomChar() {
    int type = rand() % 3;
    
    if (type == 0) {
        return 'a' + (rand() % 26); // lowercase letter
    } else if (type == 1) {
        return 'A' + (rand() % 26); // uppercase letter
    } else {
        return '0' + (rand() % 10); // digit
    }
}

// Function to generate a unique string
void generateUniqueString(char *result, char **existing, int count){ 
    while (1) {
        int middle_length = rand() % (MAX_MIDDLE_LENGTH + 1);
        
        strcpy(result, PREFIX);
        
        // Generate middle part
        for (int i = 0; i < middle_length; i++) {
            char c = getRandomChar();
            result[strlen(PREFIX) + i] = c;
        }
        
        // Add null terminator after middle part
        result[strlen(PREFIX) + middle_length] = '\0';
        
        // Append suffix
        strcat(result, SUFFIX);
        
        // Check if string already exists
        bool is_unique = true;
        for (int i = 0; i < count; i++) {
            if (strcmp(result, existing[i]) == 0) {
                is_unique = false;
                break;
            }
        }
        
        if (is_unique) {
            return;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <list_length>\n", argv[0]);
        return 1;
    }
    
    int list_length = atoi(argv[1]);
    if (list_length <= 0) {
        printf("List length must be a positive integer.\n");
        return 1;
    }
    
    // Seed random number generator
    srand(time(NULL));
    
    // Allocate memory for storing generated strings
    char **strings = malloc(list_length * sizeof(char*));
    if (!strings) {
        printf("Memory allocation failed.\n");
        return 1;
    }
    
    for (int i = 0; i < list_length; i++) {
        strings[i] = malloc(MAX_STRING_LENGTH * sizeof(char));
        if (!strings[i]) {
            printf("Memory allocation failed.\n");
            // Clean up previously allocated memory
            for (int j = 0; j < i; j++) {
                free(strings[j]);
            }
            free(strings);
            return 1;
        }
    }
    
    // Generate unique strings
    for (int i = 0; i < list_length; i++) {
        generateUniqueString(strings[i], strings, i);
    }
    
    // Write strings to file
    FILE *file = fopen("dataset.txt", "w");
    if (!file) {
        printf("Failed to open file for writing.\n");
        // Clean up allocated memory
        for (int i = 0; i < list_length; i++) {
            free(strings[i]);
        }
        free(strings);
        return 1;
    }
    
    for (int i = 0; i < list_length; i++) {
        fprintf(file, "%s", strings[i]);
        if (i < list_length - 1) {
            fprintf(file, ",");
        }
    }
    
    fclose(file);
    
    // Clean up allocated memory
    for (int i = 0; i < list_length; i++) {
        free(strings[i]);
    }
    free(strings);
    
    printf("Dataset with %d unique strings has been generated in 'dataset.txt'\n", list_length);
    
    return 0;
}