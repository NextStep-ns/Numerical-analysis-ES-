clear all; clc;

%----------------------------- EJERCICIO1 ---------------------------------
%--------------------------------------------------------------------------

%Q1
f = @(x) (exp(2*x - x.^2 - 3/4) - ((2/3)*x.^2) + 3*x - 4);
dfdt = @(x) (2-2*x)* exp(2*x - x.^2 - 3/4) - ((4/3)*x) + 3;
dfdt2 = @(x) -2 *exp(- (3/4) + 2 * x - x.^2 ) + ((2-2*x)^2)* exp(2*x - x.^2 - 3/4) -(4/3);

% No es una raiz simple porque df es egual a 0 a un momento en el interval
% entonces no es monotona.

a = 0;
b = 2;

x_vals = linspace(a, b, 1000);
y_vals = f(x_vals);

figure;
plot(x_vals, y_vals);
xlabel('x');
ylabel('f(x)');
title('Gráfico de la función f(x)');
grid on;

%Q2
x0 = 1;

[intervalo, iteraciones,c] = bisection_method(f, 0, 2, 5);
[iteraciones_newton, solucion_newton] = newton_raphson(f, dfdt,dfdt2, x0, 10^-4, 5);
disp(["iteraciones_newton : ", iteraciones_newton]);
disp(["solucion_newton : ", solucion_newton]);


%----------------------------- EJERCICIO2 ---------------------------------
%--------------------------------------------------------------------------

% Definición de la ecuación diferencial
f = @(t, y) (sin(y.^2 + t*y-t.^2))/(y.^2 - t.^2);

% Condiciones iniciales
y0 = sqrt(pi);

% Intervalo de tiempo
t_span = [0, 5];

% Número de pasos de tiempo
n_values = [10, 60];

for i = 1:length(n_values)
    n = n_values(i);
    disp(['n= ', num2str(n)]);
    %PREGUNTA1 ------------------------------------------------------------

    % Método de Euler
    t_euler = linspace(t_span(1), t_span(2), n + 1);
    y_euler = euler_method(f, t_euler, y0);

    % Método de Euler Modificado (Heun)
    t_heun = linspace(t_span(1), t_span(2), n + 1);
    y_heun = heun_method(f, t_heun, y0);

    % Método de Runge-Kutta de orden 4
    t_rk4 = linspace(t_span(1), t_span(2), n + 1);
    y_rk4 = runge_kutta_4th_order(f, t_rk4, y0);

    % Graficar soluciones
    figure;
    
    subplot(2, 1, 1);
    plot(t_euler, y_euler, '-o', 'DisplayName', 'Euler');
    hold on;
    plot(t_heun, y_heun, '-o', 'DisplayName', 'Heun');
    plot(t_rk4, y_rk4, '-o', 'DisplayName', 'RK4');
    title(['Soluciones numéricas para n = ' num2str(n)]);
    xlabel('t');
    ylabel('y');
    grid on;
    legend;
    hold off;

    %PREGUNTA2 ------------------------------------------------------------

    % Calcular y graficar errores
    t_exact_rk4 = linspace(t_span(1), t_span(2), n + 1);
    y_exact_rk4 = runge_kutta_4th_order(f, t_exact_rk4, y0);

    % Calcular errores
    error_euler = abs(interp1(t_rk4, y_rk4, t_euler, 'linear', 'extrap') - y_euler);    
    error_heun = abs(interp1(t_rk4, y_rk4, t_heun, 'linear', 'extrap') - y_heun);

    % Graficar errores
    subplot(2, 1, 2);
    plot(t_euler, error_euler, '-o', 'DisplayName', 'Euler');
    hold on;
    plot(t_heun, error_heun, '-o', 'DisplayName', 'Heun');
    title(['Errores para n = ' num2str(n)]);
    xlabel('t');
    ylabel('Error');
    grid on;
    legend;

    %PREGUNTA4 ------------------------------------------------------------

    h = (5) / (n - 1);
    Euler = h / 3 * (y_euler(1) + 4 * sum(y_euler(2:2:end-1)) + 2 * sum(y_euler(3:2:end-2)) + y_euler(end));
    Heun = h / 3 * (y_heun(1) + 4 * sum(y_heun(2:2:end-1)) + 2 * sum(y_heun(3:2:end-2)) + y_heun(end));

    disp(['Integral utilizando Simpson compuesto con Euler = ', num2str(Euler)]);
    disp(['Integral utilizando Simpson compuesto con Heun = ', num2str(Heun)]);
end

%--------------------------------------------------------------------------
%------------------------------ FUNCIONES ---------------------------------
%--------------------------------------------------------------------------

function y = euler_method(f, t, y0)
    h = t(2) - t(1);
    y = zeros(size(t));
    y(1) = y0;

    for i = 1:length(t) - 1
        y(i + 1) = y(i) + h * f(t(i), y(i));
    end
end

%--------------------------------------------------------------------------

function y = heun_method(f, t, y0)
    h = t(2) - t(1);
    y = zeros(size(t));
    y(1) = y0;

    for i = 1:length(t) - 1
        k1 = f(t(i), y(i));
        k2 = f(t(i) + h, y(i) + h * k1);
        y(i + 1) = y(i) + (h / 2) * (k1 + k2);
    end
end

%--------------------------------------------------------------------------

function y = runge_kutta_4th_order(f, t, y0)
    h = t(2) - t(1);
    y = zeros(size(t));
    y(1) = y0;

    for i = 1:length(t) - 1
        k1 = f(t(i), y(i));
        k2 = f(t(i) + h/2, y(i) + h/2 * k1);
        k3 = f(t(i) + h/2, y(i) + h/2 * k2);
        k4 = f(t(i) + h, y(i) + h * k3);
        y(i + 1) = y(i) + (h / 6) * (k1 + 2*k2 + 2*k3 + k4);
    end
end

%--------------------------------------------------------------------------

% Función de Newton-Raphson
function [iteraciones, solucion] = newton_raphson(f, df,dfdt2, x0, error, iteracion)
    x = x0;
    for iteraciones = 1:iteracion % Límite máximo de iteraciones para evitar bucles infinitos
        x1 = x - (f(x)*df(x)) / (df(x)^2 - f(x)*dfdt2(x));
        if abs(x1-x) < error
            solucion = x1;
            return;
        end
        x=x1;
    end
    
    % Si no converge, devuelve NaN
    iteraciones = NaN;
    solucion = NaN;
end

%--------------------------------------------------------------------------

% Función de Bisección
function [intervalo, iteraciones,c] = bisection_method(f, a, b, N)
    intervalo = [a, b];
    iteraciones = 0;
    
    if f(a) * f(b) >= 0
        fprintf('Bolzano is not fulfilled in the given interval.\n');
        c = [];
        return;
    end
    
    for i = 1:N
        c = (a + b) / 2;
        if f(c) * f(a) < 0
            b = c;
        else
            a = c;
        end
        iteraciones = i;
    end
    disp(["c = ", c]);
    intervalo = [a,b];
end
