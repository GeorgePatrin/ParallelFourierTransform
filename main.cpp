#include "Backend.h"
#include <time.h>


int main() {
    path input_file = "C:\\Users\\Georgiy\\Desktop\\input.raw";
    path output_file = "C:\\Users\\Georgiy\\Desktop\\output.raw";

    omp_set_num_threads(NUM_THREADS);

    // расчётная область по времени
    double rborder = 5;
    double lborder = -rborder;
    int points_num = 10000;
    MyVector<double> grid = GridFunctions::linspace(lborder, rborder, points_num);
    
    // зашумлённый сигнал (создаётся и выводится в файл)
    double amplitude = 1.0;
    MyVector<Complex> pulse = GridFunctions::pulse_envelope(grid, amplitude, 1.0, 2.0, 1.0);
    GridFunctions::make_signal_noisy(pulse, amplitude / 100);
    pulse.write_to_file(input_file);
    
    // сигнал считывается из файла
    MyVector<Complex> new_pulse{ };
    new_pulse.parallel_read_from_file(input_file);

    clock_t start = clock();
    // спектральная фильтрация
    MyVector<Complex> fouriered_pulse = FourierTransform::dft(new_pulse);
    MyVector<double> puls_spectrum = fouriered_pulse.abs();
    int peak_coord = puls_spectrum.argmax();
    GridFunctions::CutThePeak<Complex>(fouriered_pulse, peak_coord, points_num / 10);
    MyVector<Complex> clear_pulse = FourierTransform::dft(new_pulse);
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds);

    // вывод результатов в файл
    clear_pulse.write_to_file(output_file);

    return 0;
}