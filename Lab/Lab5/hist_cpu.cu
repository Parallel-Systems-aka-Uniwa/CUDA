#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define SIZE    (100*1024*1024)

void* big_random_block( int size ) {
    unsigned char *data = (unsigned char*)malloc( size );
    for (int i=0; i<size; i++)
        data[i] = rand();
    return data;
}


int main( void ) {
    unsigned char *buffer =
                     (unsigned char*)big_random_block( SIZE );

    // capture the start time
    clock_t         start, stop;
    start = clock();
    unsigned int    histo[256];
    for (int i=0; i<256; i++)
        histo[i] = 0;
    for (int i=0; i<SIZE; i++)
        histo[buffer[i]]++;
    stop = clock();
    float   elapsedTime = (float)(stop - start) /
		(float)CLOCKS_PER_SEC * 1000.0f;
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    long histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += histo[i];
    }
    printf( "Histogram Sum:  %ld\n", histoCount );

    free( buffer );
    return 0;
}
