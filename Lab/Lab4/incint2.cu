__global__ void add(int *a) {
	*a = *a + 1;
}

int main(void) {
	int a; 		// αντίγραφο του a για τον host
	int *d_a; 	// αντίγραφο του a για την device
	int size = sizeof(int);
	// Εκχώρηση μνήμης για το αντίγραφο a στην συσκευή
	cudaMalloc((void **)&d_a, size); 
	// Ορισμός αρχικής τιμής
	a = 2;
	// Αντιγραφή από host σε device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	// Κλήση add() (συνάρτηση πυρήνα) στην GPU
	add<<<1,1>>>(d_a);
	// Αντιγραφή από device σε host
	cudaMemcpy(&a, d_a, size, cudaMemcpyDeviceToHost);
	// Εκκαθάριση / απελευθέρωση μνήμης
	cudaFree(d_a);
	return 0;
}
