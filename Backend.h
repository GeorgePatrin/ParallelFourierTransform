#pragma once

#define _USE_MATH_DEFINES
#define NUM_THREADS 3
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <complex>
#include <random>
#include <cmath>
#include <math.h>

using namespace std;
using namespace filesystem;

typedef complex<double> Complex;

template<typename T>
concept comparable = requires(T v1, T v2) {
    v1 > v2;
};

template<typename T>
concept metric = requires(T v1) {
    std::abs(v1);
};

template<typename T>
class MyVector{
private:
    int size;
    T* data_array;

public:
    MyVector()=default;

    MyVector(int elems_num) {
        size = elems_num;
        data_array = new T[elems_num];
    }

    MyVector(const MyVector& other) {
        size = other.size;
        data_array = new T[size];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < size; i++) {
            data_array[i] = other.data_array[i];
        }
    }

    MyVector(MyVector&& other) noexcept {
        size = other.size;
        swap(data_array, other.data_array);
    }

    MyVector& operator=(const MyVector& other) {
        size = other.size;
        data_array = new T[size];

#pragma omp parallel for schedule(static)
        for (int i = 0; i < size; i++){
            data_array[i] = other.data_array[i];
        }
        delete[] other.data_array;
        return *this;
    }

    MyVector& operator=(MyVector&& other) noexcept {
        size = other.size;
        swap(data_array, other.data_array);
        return *this;
    }

    bool operator==(MyVector& other) {
        if (size != other.size)
            return false;
        bool is_similar = true;
#pragma omp parallel for schedule(static) reduction(&&:is_similar)
        for (int i = 0; i < size; i++){
            is_similar = (data_array[i] == other.data_array[i]);
        }
        return is_similar;
    }

    T& operator[](int i) {
        return data_array[i];
    }

    static MyVector zeros(int elems_num) {
        MyVector<T> result(elems_num);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < elems_num; i++) {
            result[i] = T{ };
        }
        return result;
    }

    void write_to_file(path file_path) {
        ofstream output_file(file_path, ios_base::binary);
        if (!output_file.good()) // Обрабатываем случай, когда файл не удалось создать или открыть
            throw runtime_error("File creating error");

        output_file.write(reinterpret_cast<const char*>(&data_array[0]), sizeof(T) * size);
        output_file.close();
    }

    void read_from_file(path file_path) {
        ifstream input_file(file_path, ios_base::binary);
        if (!input_file.good()) // Обрабатываем случай, когда файл не удалось открыть
            throw runtime_error("File open error");

        // Определяем количество элементов в считываемом массиве
        input_file.seekg(0, ios_base::end);
        int file_size = input_file.tellg();
        int elems_num = file_size / sizeof(T);
        input_file.seekg(0, ios_base::beg);

        // Итоговый массив, в который будут считываться данные
        size = elems_num;
        delete[] data_array;
        data_array = new T[size];

        // Считывание
        input_file.read(reinterpret_cast<char*>(data_array), file_size);
        input_file.close();
    }

    void parallel_read_from_file(path file_path) {
        ifstream input_file(file_path, ios_base::binary);
        if (!input_file.good()) // Обрабатываем случай, когда файл не удалось открыть
            throw runtime_error("File open error");

        // Определяем количество элементов в считываемом массиве
        input_file.seekg(0, ios_base::end);
        int file_size = input_file.tellg();
        int elems_num = file_size / sizeof(T);
        input_file.seekg(0, ios_base::beg);

        // Итоговый массив, в который будут считываться данные
        size = elems_num;
        delete[] data_array;
        data_array = new T[size];

        // В каждом потоке считываем свой кусок файла
        int num_threads = NUM_THREADS;
        int frame_size = (elems_num % num_threads == 0) ?
            elems_num / num_threads :
            (elems_num / num_threads + 1);
        input_file.close();
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int current_start = frame_size * thread_id;
            int current_size = (thread_id == (num_threads - 1)) ?
                (file_size - current_start * sizeof(T)) :
                frame_size * sizeof(T);

            ifstream input_file(file_path, ios_base::binary);
            input_file.seekg(current_start * sizeof(T));
            input_file.read(reinterpret_cast<char*>(&data_array[current_start]), current_size);
            input_file.close();
        }
    }

    int argmax() requires comparable<T> {
        if (size == 0)
            throw runtime_error("Array should be not empty");

        // Создаём задачу для каждого потока
        int num_threads = NUM_THREADS;
        int frame_size = (size % num_threads == 0) ?
            size / num_threads :
            size / num_threads + 1;
        MyVector<int> starts(num_threads);
        MyVector<int> ends(num_threads);
        for (int i = 0; i < num_threads; i++) {
            starts[i] = frame_size * i;
            ends[i] = min(frame_size * (i + 1), size);
        }

        // В каждом потоке считываем свой кусок файла
        MyVector<int> argmaxes(num_threads);
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int local_record = starts[thread_id];
            double local_max = data_array[local_record];
            for (int i = starts[thread_id] + 1; i < ends[thread_id]; i++) {
                if (data_array[i] > local_max) {
                    local_record = i;
                    local_max = data_array[i];
                }
            }
            argmaxes[thread_id] = local_record;
        }

        //По всем потокам выбираем argmax
        int total_record = argmaxes[0];
        double total_max = data_array[total_record];
        for (int i = 1; i < num_threads; i++) {
            if (data_array[argmaxes[i]] > total_max) {
                total_record = argmaxes[i];
                total_max = data_array[argmaxes[i]];
            }
        }
        return total_record;
    }

    MyVector<double> abs() requires metric<T> {
        if (size == 0)
            throw runtime_error("Array should be not empty");

        MyVector<double> abs_vector(size);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < size; i++) {
            abs_vector[i] = std::abs(data_array[i]);
        }
        return abs_vector;
    }

    int get_size() const {
        return size;
    }

    void print() {
        for (int i = 0; i < size; i++)
            cout << data_array[i] << ' ';
        cout << endl;
    }

    ~MyVector() {
        delete[] data_array;
    }
};



namespace GridFunctions {
    MyVector<double> linspace(double left_border, double right_border, int points_num) {
        if (points_num < 2)
            throw runtime_error("points_num should be graiter, then 2");
        if (left_border >= right_border)
            throw runtime_error("incorrect interval");

        double step_size = (right_border - left_border) / (points_num - 1);
        MyVector<double> grid_vector(points_num);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < points_num; i++) {
            grid_vector[i] = left_border + step_size * i;
        }

        return grid_vector;
    }

    MyVector<Complex> pulse_envelope(MyVector<double>& time,
        double amplitude = 1.0, double chirp = 1.0,
        double width = 1.0, double phase_shift = 1.0) {

        int data_size = time.get_size();
        MyVector<Complex> solitary_pulse(data_size);
#pragma omp parallel for schedule(dynamic)
        for (int n = 0; n < data_size; n++)
            solitary_pulse[n] = amplitude /
            pow(cosh(time[n] * 2.65 / width), 1.0 - 1i * chirp) *
            exp(1i * phase_shift);
        return solitary_pulse;
    }

    void make_signal_noisy(MyVector<Complex>& signal, double noise_level) {
        int data_size = signal.get_size();
        double distrib_border = noise_level / sqrt(2); // Чтобы у комплексного шума была амплитуда по модулю
        random_device rd; // Источник случайных чисел
        mt19937 gen(rd()); // Инициализация генератора Мерсенна

        //Создание объекта распределения и генерация случайных чисел
        uniform_real_distribution<double> distribution{ };
#pragma omp parallel for schedule(static) private(distribution, gen)
        for (int i = 0; i < data_size; i++) {
            double real_part = distribution(gen) * 2 * distrib_border - distrib_border;
            double imag_part = distribution(gen) * 2 * distrib_border - distrib_border;
            signal[i] += Complex(real_part, imag_part);
        }
    }

    template<typename T>
    void CutThePeak(MyVector<T>& signal, int peak_coord, int neighborhood) {
        int left_idx = max(0, peak_coord - neighborhood / 2);
        int right_idx = min(peak_coord + neighborhood / 2, signal.get_size() - 1);

        for (int i = 0; i < left_idx; i++)
            signal[i] = T{ };

        for (int i = right_idx; i < signal.get_size(); i++)
            signal[i] = T{ };
    }
}



namespace FourierTransform {
    MyVector<Complex> dft(MyVector<Complex>& func_vector) {
        int points_num = func_vector.get_size();

        // Умножение вектора на матрицу
        MyVector<Complex> transform_result = MyVector<Complex>::zeros(points_num);
#pragma omp parallel for schedule(dynamic) collapse(2)
        for (int n = 0; n < points_num; n++) {
            Complex val_n = func_vector[n];  // минимизируем обращения к памяти
            for (int k = 0; k < points_num; k++) {
                double current_frequency = (double)k * n / points_num;
                transform_result[k] += exp(-2.0i * M_PI * current_frequency) * val_n;
            }
        }

        return transform_result;
    }

    MyVector<double> dft_freq(MyVector<double>& time_vector) {
        int points_num = time_vector.get_size();
        // считаем, что массив времени отсортирован
        double time_area = time_vector[points_num - 1] - time_vector[points_num];
        double nyq_freq = 0.5 * points_num / time_area;

        MyVector<double> freqs_vector = GridFunctions::linspace(-nyq_freq, nyq_freq, points_num);
        return freqs_vector;
    }
}
