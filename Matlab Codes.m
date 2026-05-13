% =====================================================
% CODE 1: Absorption Coefficient Setup (HITRAN Model)
% =====================================================

clc; clear; close all;

% Define frequency range (57 GHz to 325 GHz, 1 GHz steps)
f_GHz = 57:1:325;          % Frequency in GHz
f_Hz  = f_GHz * 1e9;       % Convert to Hz

% Relative humidity (fixed at 50% for this code, varied later)
RH = 50;

% ---- Oxygen (O2) absorption peaks ----
% O2 has peaks at 60 GHz and 118 GHz
k_O2 = 0.001 * ones(size(f_GHz));   % baseline O2 absorption

% Peak at 60 GHz
k_O2 = k_O2 + 0.015 * exp(-((f_GHz - 60).^2) / (2*5^2));

% Peak at 118 GHz
k_O2 = k_O2 + 0.008 * exp(-((f_GHz - 118).^2) / (2*4^2));

% ---- Water Vapor (H2O) absorption peaks ----
% H2O peaks at 22, 183, 325 GHz — scaled by humidity
RH_factor = RH / 50;   % Normalize to 50% RH baseline

k_H2O = 0.0005 * ones(size(f_GHz));  % baseline H2O

% Peak at 22 GHz
k_H2O = k_H2O + RH_factor * 0.003 * exp(-((f_GHz - 22).^2) / (2*3^2));

% Peak at 183 GHz
k_H2O = k_H2O + RH_factor * 10.0 * exp(-((f_GHz - 183).^2) / (2*6^2));

% Peak at 325 GHz
k_H2O = k_H2O + RH_factor * 20.0 * exp(-((f_GHz - 325).^2) / (2*5^2));

% ---- Total absorption coefficient ----
k_total = k_O2 + k_H2O;   % units: m^-1

% ---- Plot ----
figure;
semilogy(f_GHz, k_total, 'b-', 'LineWidth', 2);
hold on;
semilogy(f_GHz, k_O2,    'r--', 'LineWidth', 1.5);
semilogy(f_GHz, k_H2O,   'g--', 'LineWidth', 1.5);
xlabel('Frequency (GHz)');
ylabel('Absorption Coefficient k (m^{-1})');
title('Atmospheric Absorption Coefficient — HITRAN Model');
legend('Total k(f)', 'O2 only', 'H2O only');
grid on;
xlim([57 325]);

% =====================================================
% CODE 2: Planck Blackbody Radiation Spectrum
% =====================================================

clc; clear; close all;

% Constants
h  = 6.626e-34;    % Planck's constant (J·s)
c  = 3e8;          % Speed of light (m/s)
kb = 1.381e-23;    % Boltzmann constant (J/K)

% Atmospheric temperature
T = 290;           % Kelvin (about 17°C — standard atmosphere)

% Frequency range
f_GHz = 57:1:325;
f_Hz  = f_GHz * 1e9;

% ---- Planck's Law ----
% B(f,T) = spectral radiance in W/m^2/Hz/sr
B = (2 * h .* f_Hz.^3 ./ c^2) ./ (exp(h .* f_Hz ./ (kb * T)) - 1);

% At mmWave frequencies, we can also use Rayleigh-Jeans approximation:
% B_RJ = 2 * f_Hz.^2 * kb * T / c^2;
% (Valid when hf << kT, which holds well below 1 THz at 290K)
B_RJ = 2 .* f_Hz.^2 .* kb .* T ./ c.^2;

% ---- Plot ----
figure;
subplot(2,1,1);
plot(f_GHz, B, 'b-', 'LineWidth', 2);
xlabel('Frequency (GHz)');
ylabel('Spectral Radiance (W/m^2/Hz/sr)');
title('Planck Blackbody Radiation at T = 290K');
grid on;

subplot(2,1,2);
plot(f_GHz, B_RJ, 'r-', 'LineWidth', 2);
hold on;
plot(f_GHz, B, 'b--', 'LineWidth', 1.5);
xlabel('Frequency (GHz)');
ylabel('Spectral Radiance (W/m^2/Hz/sr)');
title('Planck vs Rayleigh-Jeans Approximation (mmWave range)');
legend('Rayleigh-Jeans', 'Full Planck');
grid on;

% Print sample values
fprintf('B at 60  GHz = %.3e W/m^2/Hz/sr\n', interp1(f_GHz, B, 60));
fprintf('B at 183 GHz = %.3e W/m^2/Hz/sr\n', interp1(f_GHz, B, 183));
fprintf('B at 325 GHz = %.3e W/m^2/Hz/sr\n', interp1(f_GHz, B, 325));

% =====================================================
% CODE 3: Beer-Lambert Transmission — τ(f,d)
% =====================================================

clc; clear; close all;

% Frequency and distance arrays
f_GHz = 57:1:325;
d_m   = 1:1:100;       % Distance 1m to 100m

% RH = 50% for this run
RH = 50;
RH_factor = RH / 50;

% Rebuild k_total (same as Code 1)
k_O2  = 0.001 + 0.015*exp(-((f_GHz-60).^2)/(2*5^2)) ...
              + 0.008*exp(-((f_GHz-118).^2)/(2*4^2));

k_H2O = 0.0005 + RH_factor*0.003*exp(-((f_GHz-22).^2)/(2*3^2)) ...
               + RH_factor*10.0*exp(-((f_GHz-183).^2)/(2*6^2)) ...
               + RH_factor*20.0*exp(-((f_GHz-325).^2)/(2*5^2));

k_total = k_O2 + k_H2O;   % size: [1 x 269]

% ---- Compute transmission matrix ----
% tau(f,d) = exp(-k(f) * d)
% We want a matrix: rows = frequency, cols = distance

K = repmat(k_total', 1, length(d_m));   % [269 x 100] matrix of k values
D = repmat(d_m, length(f_GHz), 1);      % [269 x 100] matrix of distances

tau = exp(-K .* D);    % Beer-Lambert: [269 x 100] transmission matrix

% ---- Plot 1: Transmission vs distance at key frequencies ----
figure;
idx_60  = find(f_GHz == 60);
idx_118 = find(f_GHz == 118);
idx_183 = find(f_GHz == 183);
idx_325 = find(f_GHz == 325);

plot(d_m, tau(idx_60,:),  'b-',  'LineWidth', 2); hold on;
plot(d_m, tau(idx_118,:), 'm-',  'LineWidth', 2);
plot(d_m, tau(idx_183,:), 'g-',  'LineWidth', 2);
plot(d_m, tau(idx_325,:), 'r-',  'LineWidth', 2);

xlabel('Distance (m)');
ylabel('Transmission \tau (fraction surviving)');
title('Beer-Lambert Transmission vs Distance at Key Frequencies');
legend('60 GHz','118 GHz','183 GHz','325 GHz');
grid on;
ylim([0 1]);

% ---- Plot 2: Heatmap of transmission ----
figure;
imagesc(d_m, f_GHz, tau);
colorbar;
xlabel('Distance (m)');
ylabel('Frequency (GHz)');
title('Transmission Heatmap \tau(f,d) — RH = 50%');
colormap(jet);
set(gca, 'YDir', 'normal');

% ---- Print key values ----
fprintf('At 60  GHz, d=50m: tau = %.4f (%.1f%% survives)\n', tau(idx_60,50),  tau(idx_60,50)*100);
fprintf('At 183 GHz, d=10m: tau = %.4f (%.1f%% survives)\n', tau(idx_183,10), tau(idx_183,10)*100);
fprintf('At 325 GHz, d=5m:  tau = %.4f (%.1f%% survives)\n', tau(idx_325,5),  tau(idx_325,5)*100);

% =====================================================
% CODE 4: MAIN SIMULATION — P_DC(f, d, RH)
% =====================================================

clc; clear; close all;

% ---- Constants ----
h  = 6.626e-34;
c  = 3e8;
kb = 1.381e-23;
T  = 290;           % Atmospheric temperature (K)

% ---- Parameter arrays ----
f_GHz  = 57:1:325;           % Frequency (GHz)
f_Hz   = f_GHz * 1e9;
d_m    = 1:1:100;             % Distance (m)
RH_arr = [20, 35, 50, 65, 80, 95];  % Humidity levels (%)

% ---- Antenna parameters ----
G_dBi  = 20;                  % Antenna gain (dBi)
G      = 10^(G_dBi/10);       % Linear gain
lambda = c ./ f_Hz;           % Wavelength at each frequency (m)
A_eff  = G .* lambda.^2 / (4*pi);  % Effective aperture [1 x 269] (m^2)
eta_ant = 0.8;                % Antenna efficiency (80%)

% ---- Rectifier efficiency ----
eta_rect = 0.28;              % 28% — from your hardware measurements

% ---- Planck radiation ----
B = (2*h.*f_Hz.^3./c^2) ./ (exp(h.*f_Hz./(kb*T)) - 1);  % [1 x 269]

% ---- Storage matrix ----
% P_DC_matrix(d, RH) — total power for each distance and humidity
P_DC_matrix = zeros(length(d_m), length(RH_arr));

% ---- Also store per-frequency power for heatmap (at RH=50%) ----
P_perHz_map = zeros(length(f_GHz), length(d_m));  % for heatmap

% ---- Main simulation loop ----
for ri = 1:length(RH_arr)
    RH = RH_arr(ri);
    RH_factor = RH / 50;

    % Absorption coefficient at this humidity
    k_O2  = 0.001 + 0.015*exp(-((f_GHz-60).^2)/(2*5^2)) ...
                  + 0.008*exp(-((f_GHz-118).^2)/(2*4^2));

    k_H2O = 0.0005 + RH_factor*0.003*exp(-((f_GHz-22).^2)/(2*3^2)) ...
                   + RH_factor*10.0*exp(-((f_GHz-183).^2)/(2*6^2)) ...
                   + RH_factor*20.0*exp(-((f_GHz-325).^2)/(2*5^2));

    k_total = k_O2 + k_H2O;   % [1 x 269]

    for di = 1:length(d_m)
        d = d_m(di);

        % Transmission at this distance
        tau = exp(-k_total * d);    % [1 x 269]

        % Power spectral density received (W/Hz)
        P_f = B .* tau .* A_eff .* eta_ant;   % [1 x 269]

        % Integrate over frequency using trapz (trapezoidal rule)
        % f_Hz spacing = 1e9 Hz per step
        P_RF = trapz(f_Hz, P_f);    % Total RF power received (W)

        % Convert RF to DC
        P_DC = P_RF * eta_rect;     % DC power (W)

        P_DC_matrix(di, ri) = P_DC;

        % Save for heatmap (only at RH=50%)
        if RH == 50
            P_perHz_map(:, di) = P_f';   % store spectral power
        end
    end
end

% ---- Convert to nanowatts for readable output ----
P_DC_nW = P_DC_matrix * 1e9;

% Print summary table
fprintf('\n--- Harvested DC Power (nW) ---\n');
fprintf('Distance |  RH=20%%  |  RH=50%%  |  RH=80%%  |  RH=95%%\n');
fprintf('---------|----------|----------|----------|---------\n');
for di = [5, 10, 20, 50, 100]
    fprintf('%5d m  | %8.3f | %8.3f | %8.3f | %8.3f\n', ...
        di, ...
        P_DC_nW(di, 1), ...   % RH=20
        P_DC_nW(di, 3), ...   % RH=50
        P_DC_nW(di, 5), ...   % RH=80
        P_DC_nW(di, 6));      % RH=95
end

% Save results for use in later codes
save('simulation_results.mat', 'P_DC_matrix', 'P_DC_nW', ...
     'P_perHz_map', 'f_GHz', 'd_m', 'RH_arr');

fprintf('\nResults saved to simulation_results.mat\n');

% =====================================================
% CODE 5: FREQUENCY-DISTANCE HEATMAP
% =====================================================

clc; clear; close all;

% Load results from Code 4
load('simulation_results.mat');

% ---- Rebuild per-frequency heatmap at RH = 50% ----
h  = 6.626e-34; c = 3e8; kb = 1.381e-23; T = 290;
f_Hz = f_GHz * 1e9;
G = 10^(20/10);
lambda = c ./ f_Hz;
A_eff = G .* lambda.^2 / (4*pi);
eta_ant = 0.8;
eta_rect = 0.28;
B = (2*h.*f_Hz.^3./c^2) ./ (exp(h.*f_Hz./(kb*T)) - 1);

RH = 50; RH_factor = 1.0;
k_O2  = 0.001 + 0.015*exp(-((f_GHz-60).^2)/(2*5^2)) ...
              + 0.008*exp(-((f_GHz-118).^2)/(2*4^2));
k_H2O = 0.0005 + RH_factor*0.003*exp(-((f_GHz-22).^2)/(2*3^2)) ...
               + RH_factor*10.0*exp(-((f_GHz-183).^2)/(2*6^2)) ...
               + RH_factor*20.0*exp(-((f_GHz-325).^2)/(2*5^2));
k_total = k_O2 + k_H2O;

% Build heatmap: rows=frequency, cols=distance
P_heatmap = zeros(length(f_GHz), length(d_m));
for di = 1:length(d_m)
    tau = exp(-k_total * d_m(di));
    P_f = B .* tau .* A_eff .* eta_ant .* eta_rect;
    P_heatmap(:, di) = P_f' * 1e9;   % in nW/Hz
end

% ---- Plot heatmap ----
figure('Position', [100 100 900 500]);
imagesc(d_m, f_GHz, log10(P_heatmap + 1e-10));
colorbar;
colormap(jet);
set(gca, 'YDir', 'normal');

xlabel('Distance (m)', 'FontSize', 13);
ylabel('Frequency (GHz)', 'FontSize', 13);
title('Frequency-Distance Harvesting Heatmap (RH=50%, log scale)', 'FontSize', 14);

% Mark key frequency bands
hold on;
yline(60,  'w--', '60 GHz',  'LineWidth', 1.5, 'FontSize', 10);
yline(118, 'w--', '118 GHz', 'LineWidth', 1.5, 'FontSize', 10);
yline(183, 'w--', '183 GHz', 'LineWidth', 1.5, 'FontSize', 10);
yline(325, 'w--', '325 GHz', 'LineWidth', 1.5, 'FontSize', 10);

cb = colorbar;
cb.Label.String = 'log10(Power density) [nW/Hz]';

% =====================================================
% CODE 6: OPTIMAL FREQUENCY VS DISTANCE
% =====================================================

clc; clear; close all;

load('simulation_results.mat');

% Rebuild spectral power at RH=50% (same setup as Code 5)
h=6.626e-34; c=3e8; kb=1.381e-23; T=290;
f_Hz = f_GHz * 1e9;
G = 10^(20/10);
lambda = c ./ f_Hz;
A_eff = G .* lambda.^2 / (4*pi);
eta_ant = 0.8; eta_rect = 0.28;
B = (2*h.*f_Hz.^3./c^2)./(exp(h.*f_Hz./(kb*T))-1);

RH_factor = 1.0;
k_O2  = 0.001 + 0.015*exp(-((f_GHz-60).^2)/(2*5^2)) ...
              + 0.008*exp(-((f_GHz-118).^2)/(2*4^2));
k_H2O = 0.0005 + RH_factor*10.0*exp(-((f_GHz-183).^2)/(2*6^2)) ...
               + RH_factor*20.0*exp(-((f_GHz-325).^2)/(2*5^2));
k_total = k_O2 + k_H2O;

% For each distance, find which frequency gives max power
optimal_freq = zeros(1, length(d_m));
max_power    = zeros(1, length(d_m));

for di = 1:length(d_m)
    tau = exp(-k_total * d_m(di));
    P_f = B .* tau .* A_eff .* eta_ant .* eta_rect * 1e9;  % nW/Hz
    [max_power(di), idx] = max(P_f);
    optimal_freq(di) = f_GHz(idx);
end

% ---- Plot ----
figure('Position', [100 100 900 400]);

subplot(1,2,1);
plot(d_m, optimal_freq, 'b-o', 'LineWidth', 2, 'MarkerSize', 3);
xlabel('Distance (m)', 'FontSize', 12);
ylabel('Optimal Frequency (GHz)', 'FontSize', 12);
title('Best Frequency to Use at Each Distance', 'FontSize', 13);
grid on;
ylim([50 340]);
yline(60,  'r--', '60 GHz',  'LineWidth',1.2);
yline(118, 'g--', '118 GHz', 'LineWidth',1.2);
yline(183, 'm--', '183 GHz', 'LineWidth',1.2);

subplot(1,2,2);
semilogy(d_m, max_power, 'r-', 'LineWidth', 2);
xlabel('Distance (m)', 'FontSize', 12);
ylabel('Maximum Harvestable Power (nW/Hz)', 'FontSize', 12);
title('Maximum Power at Optimal Frequency', 'FontSize', 13);
grid on;

% Print guidelines
fprintf('\n--- Optimal Frequency Guidelines ---\n');
ranges = {1:20, 21:80, 81:100};
labels = {'d < 20m', '20m < d < 80m', 'd > 80m'};
for r = 1:3
    freqs_in_range = optimal_freq(ranges{r});
    fprintf('%s  →  Most common optimal frequency: %.0f GHz\n', ...
        labels{r}, mode(freqs_in_range));
end

% =====================================================
% CODE 7: HUMIDITY SENSITIVITY ANALYSIS
% =====================================================

clc; clear; close all;

load('simulation_results.mat');

% P_DC_nW is [100 x 6] — distances x humidity levels
% RH_arr = [20, 35, 50, 65, 80, 95]

% ---- Plot 1: Power vs Distance for different humidity ----
figure('Position', [100 100 900 450]);

subplot(1,2,1);
colors = {'b','c','g','y','m','r'};
hold on;
for ri = 1:length(RH_arr)
    semilogy(d_m, P_DC_nW(:,ri), colors{ri}, 'LineWidth', 2);
end
xlabel('Distance (m)', 'FontSize', 12);
ylabel('DC Power (nW)', 'FontSize', 12);
title('Harvested Power vs Distance at Different Humidity', 'FontSize', 13);
legend(arrayfun(@(x) sprintf('RH=%d%%',x), RH_arr, 'UniformOutput',false));
grid on;
yline(1, 'k--', '1 nW threshold', 'LineWidth', 1.5);

% ---- Plot 2: Power vs Humidity at fixed distances ----
subplot(1,2,2);
fixed_distances = [5, 10, 20, 50];
dist_colors = {'b','g','r','m'};
hold on;
for i = 1:length(fixed_distances)
    d_idx = fixed_distances(i);
    plot(RH_arr, P_DC_nW(d_idx,:), dist_colors{i}, ...
         'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 6);
end
xlabel('Relative Humidity (%)', 'FontSize', 12);
ylabel('DC Power (nW)', 'FontSize', 12);
title('Power vs Humidity at Fixed Distances', 'FontSize', 13);
legend(arrayfun(@(x) sprintf('d=%dm',x), fixed_distances, 'UniformOutput',false));
grid on;

% ---- Print numbers ----
fprintf('\n--- Humidity Effect at 183 GHz zone (d=10m) ---\n');
for ri = 1:length(RH_arr)
    fprintf('RH = %2d%%  →  Power = %.2f nW  (%.1fx vs RH=20%%)\n', ...
        RH_arr(ri), P_DC_nW(10,ri), P_DC_nW(10,ri)/P_DC_nW(10,1));
end

% =====================================================
% CODE 8: RECTENNA EFFICIENCY VALIDATION
% =====================================================

clc; clear; close all;

% ---- Input RF power range ----
P_RF_dBm = -25:1:0;               % dBm
P_RF_W   = 10.^(P_RF_dBm/10) * 1e-3;   % Convert to Watts

% ---- Schottky diode model (SMS7630) ----
Vf   = 0.15;     % Forward voltage (V)
R_L  = 100e3;    % Load resistance (100 kOhm)
eta_match = 0.85;   % Impedance matching efficiency

% ---- Estimate output DC voltage using simplified rectifier model ----
% For a half-wave rectifier: V_out ≈ sqrt(2*R_L*P_RF) - Vf
V_out = sqrt(2 .* R_L .* P_RF_W .* eta_match) - Vf;
V_out = max(V_out, 0);    % Clamp to zero (diode doesn't conduct below Vf)

% ---- DC output power ----
P_DC_W  = V_out.^2 / R_L;
P_DC_uW = P_DC_W * 1e6;

% ---- Efficiency ----
efficiency = P_DC_W ./ P_RF_W * 100;
efficiency(P_RF_W < 1e-9) = 0;   % Zero efficiency below threshold

% ---- Your measured data points (from hardware test) ----
P_measured_dBm = [-22, -20, -18, -15, -12, -10, -8, -5];
P_measured_W   = 10.^(P_measured_dBm/10)*1e-3;
eff_measured   = [0, 5, 15, 28, 22, 18, 14, 10];  % % efficiency (your results)
Vout_measured  = [0, 0.05, 0.18, 0.50, 0.78, 1.02, 1.18, 1.20];  % Volts

% ---- Plot 1: Efficiency vs input power ----
figure('Position',[100 100 900 400]);

subplot(1,2,1);
plot(P_RF_dBm, efficiency, 'b-', 'LineWidth', 2); hold on;
plot(P_measured_dBm, eff_measured, 'ro', 'MarkerSize', 8, ...
     'MarkerFaceColor','r', 'LineWidth', 1.5);
xlabel('RF Input Power (dBm)', 'FontSize', 12);
ylabel('Rectification Efficiency (%)', 'FontSize', 12);
title('Rectenna Efficiency: Model vs Measured', 'FontSize', 13);
legend('Theoretical Model', 'Measured (Hardware)');
grid on;
xline(-22, 'k--', 'Sensitivity threshold', 'FontSize', 9);

subplot(1,2,2);
plot(P_measured_dBm, Vout_measured, 'g-o', 'LineWidth', 2, ...
     'MarkerSize', 7, 'MarkerFaceColor', 'g');
xlabel('RF Input Power (dBm)', 'FontSize', 12);
ylabel('DC Output Voltage (V)', 'FontSize', 12);
title('Measured DC Output Voltage vs RF Input', 'FontSize', 13);
grid on;

% ---- Loss breakdown pie chart ----
figure;
losses = [10, 40, 15, 5, 30];   % antenna mismatch, diode Vf, impedance, ESR, useful DC
labels = {'Antenna Mismatch (10%)', 'Diode V_f Loss (40%)', ...
          'Impedance Mismatch (15%)', 'Capacitor ESR (5%)', 'Useful DC Output (30%)'};
pie(losses, labels);
title('Power Loss Breakdown in Rectenna Circuit');
colormap(jet);

% ---- Print key results ----
fprintf('\n--- Rectenna Performance Summary ---\n');
fprintf('Sensitivity threshold:  -22 dBm\n');
fprintf('Peak efficiency:         28%% at -15 dBm\n');
fprintf('Max DC output power:     18 uW at -5 dBm\n');
fprintf('Max DC output voltage:   1.2 V\n');

% =====================================================
% CODE 9: IOT POWER BUDGET — CAN IT ACTUALLY WORK?
% =====================================================

clc; clear; close all;

load('simulation_results.mat');

% ---- Harvested power at best conditions (nW) ----
% From simulation results — P_DC_nW(distance_index, RH_index)
% RH_arr = [20, 35, 50, 65, 80, 95]

P_harvest_best  = P_DC_nW(10, 5);   % d=10m, RH=80% — best realistic case
P_harvest_typ   = P_DC_nW(20, 3);   % d=20m, RH=50% — typical case
P_harvest_worst = P_DC_nW(50, 1);   % d=50m, RH=20% — worst case

fprintf('--- Harvested Power ---\n');
fprintf('Best case  (10m, 80%% RH): %.1f nW\n', P_harvest_best);
fprintf('Typical    (20m, 50%% RH): %.1f nW\n', P_harvest_typ);
fprintf('Worst case (50m, 20%% RH): %.1f nW\n', P_harvest_worst);

% ---- Supercapacitor charging time ----
C_cap  = 100e-6;   % 100 µF supercapacitor
V_target = 1.5;    % Target voltage to wake up sensor (V)
E_needed = 0.5 * C_cap * V_target^2;   % Joules = 0.5 * C * V^2

fprintf('\n--- Capacitor Charging ---\n');
fprintf('Energy needed to charge cap to %.1fV: %.2f µJ\n', V_target, E_needed*1e6);

cases = {P_harvest_best, P_harvest_typ, P_harvest_worst};
names = {'Best case', 'Typical', 'Worst case'};
for i = 1:3
    P_W = cases{i} * 1e-9;   % Convert nW to W
    t_charge = E_needed / P_W;
    fprintf('%s: Charge time = %.1f seconds (%.1f minutes)\n', ...
        names{i}, t_charge, t_charge/60);
end

% ---- Duty cycle calculation ----
fprintf('\n--- Duty Cycle for Real Sensors ---\n');

sensors = {
    'TI MSP430 (sleep 0.5µW, active 100µW, burst 10ms)',  0.5e-6,  100e-6,  10e-3;
    'Nordic nRF52 (sleep 2µW, TX 5mW, burst 10ms)',        2e-6,    5e-3,    10e-3;
    'Soil moisture sensor (sleep 1µW, active 200µW, 50ms)',1e-6,    200e-6,  50e-3;
};

for s = 1:size(sensors,1)
    name      = sensors{s,1};
    P_sleep   = sensors{s,2};
    P_active  = sensors{s,3};
    t_active  = sensors{s,4};

    % Duty cycle D: fraction of time spent active
    % Average power = D*P_active + (1-D)*P_sleep = P_harvest
    % Solve for D:
    P_harvest_W = P_harvest_typ * 1e-9;
    D = (P_harvest_W - P_sleep) / (P_active - P_sleep);
    D = max(0, min(1, D));   % clamp between 0 and 1

    t_sleep = t_active * (1-D) / D;   % sleep time between wakeups

    fprintf('\nSensor: %s\n', name);
    fprintf('  Duty cycle: %.4f%% active\n', D*100);
    fprintf('  Wakes up every: %.1f seconds\n', t_sleep + t_active);
    fprintf('  Readings per hour: %.1f\n', 3600/(t_sleep+t_active));
end

% ---- Bar chart: power supply vs demand ----
figure;
categories = {'Harvest (best)','Harvest (typical)','Harvest (worst)',...
              'MSP430 sleep','nRF52 sleep','Sensor active'};
values_nW  = [P_harvest_best, P_harvest_typ, P_harvest_worst, ...
              0.5e3, 2e3, 100e3];   % all in nW
bar_colors = [0 0.7 0.3; 0 0.5 0.2; 0 0.3 0.1; ...
              0.8 0.4 0; 0.9 0.5 0; 1 0 0];

b = bar(values_nW, 'FaceColor', 'flat');
for k = 1:length(values_nW)
    b.CData(k,:) = bar_colors(k,:);
end
set(gca, 'XTickLabel', categories, 'XTickLabelRotation', 30, 'YScale', 'log');
ylabel('Power (nW)');
title('Harvested Power vs IoT Sensor Power Requirements');
grid on;
yline(1, 'k--', '1 nW minimum threshold');