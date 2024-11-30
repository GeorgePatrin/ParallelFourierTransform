#include "Backend.h"
#include <time.h>


int main() {
    path input_file = "C:\\Users\\Georgiy\\Desktop\\pulse_input.raw";
    path time_file = "C:\\Users\\Georgiy\\Desktop\\time.raw";
    path output_file = "C:\\Users\\Georgiy\\Desktop\\pulse_output.raw";

    omp_set_num_threads(NUM_THREADS);

    // расчётная область по времени
    double rborder = 5;
    double lborder = -rborder;
    int points_num = 10000 + 1;
    MyVector<double> grid = GridFunctions::linspace(lborder, rborder, points_num);
    grid.write_to_file(time_file);
    
    // зашумлённый сигнал (создаётся и выводится в файл)
    double amplitude = 1.0;
    MyVector<Complex> pulse = GridFunctions::pulse_envelope(grid, amplitude, 1.0, 2.0, 1.0);
    GridFunctions::make_signal_noisy(pulse, amplitude / 30);
    pulse.write_to_file(input_file);
    
    // сигнал считывается из файла
    MyVector<Complex> new_pulse{ };
    new_pulse.parallel_read_from_file(input_file);

    clock_t start = clock();
    // спектральная фильтрация
    MyVector<Complex> fouriered_pulse = FourierTransform::dft(new_pulse);
    MyVector<Complex> pulse_spectrum = FourierTransform::dft_shift(fouriered_pulse);
    MyVector<double> pulse_abs_spectrum = pulse_spectrum.abs();
    int peak_coord = pulse_abs_spectrum.argmax();
    GridFunctions::CutThePeak<Complex>(pulse_spectrum, peak_coord, points_num / 100);
    MyVector<Complex> cuted_pulse = FourierTransform::dft_shift(pulse_spectrum);
    MyVector<Complex> clear_pulse = FourierTransform::idft(cuted_pulse);
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds);

    // вывод результатов в файл
    clear_pulse.write_to_file(output_file);

    return 0;
}
