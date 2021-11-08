%% QDP Processing

clear all; close all; clc

%% 3.1 Device Cooling 

%%% Load the data 
load('time_sweeps.mat')


t=temperaturemeasurementTP0320211013Sweep15(:,2);
y1=temperaturemeasurementTP0320211013Sweep15(:,3).*1e3;
y2=temperaturemeasurementTP0320211013Sweep15(:,4).*1e3;

tau=zeros(2,7);
figure(1)
plot(t,y1,'*','color',[0 0.4470 0.7410])
hold on
plot(t,y2,'*','color',[0.8500    0.3250    0.0980])
plot(t,y2-y1,'color',[0.9290 0.6940 0.1250])

% Convert X and Y into a table, which is the form fitnlm() likes the input data to be in.
tbl = table(t-t(1), y1);
% Define the model as Y = a + exp(-b*x)
% Note how this "x" of modelfun is related to big X and big Y.
% x((:, 1) is actually X and x(:, 2) is actually Y - the first and second columns of the table.
modelfun = @(b,x) b(1) + b(2) * exp(-b(3)*x(:, 1));  
beta0 = [16, -0.1, 0.001]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.
% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'b-', 'LineWidth', 1);
grid on;
tau(1,1)=1/coefficients(3);

tb2 = table(t-t(1), y2);
beta0 = [23, -0.1, 0.001];
mdl = fitnlm(tb2, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.
% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'r-', 'LineWidth', 1);
grid on;

tau(2,1)=1/coefficients(3);
%%
t=temperaturemeasurementTP0320211013Sweep16(:,2);
y1=temperaturemeasurementTP0320211013Sweep16(:,3).*1e3;
y2=temperaturemeasurementTP0320211013Sweep16(:,4).*1e3;

plot(t,y1,'*','color',[0 0.4470 0.7410])
hold on
plot(t,y2,'*','color',[0.8500    0.3250    0.0980])
plot(t,y2-y1,'color',[0.9290 0.6940 0.1250])


tbl = table(t-t(1), y1);

modelfun = @(b,x) b(1) + b(2) * exp(-b(3)*x(:, 1));  
beta0 = [23, -0.1, 0.001]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);

coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'b-', 'LineWidth', 1);
grid on;
tau(1,2)=1/coefficients(3);

tb2 = table(t-t(1), y2);
beta0 = [26, -0.1, 0.001];
mdl = fitnlm(tb2, modelfun, beta0);
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'r-', 'LineWidth', 1);
grid on;

tau(2,2)=1/coefficients(3);
%%
t=temperaturemeasurementTP0320211013Sweep17(:,2);
y1=temperaturemeasurementTP0320211013Sweep17(:,3).*1e3;
y2=temperaturemeasurementTP0320211013Sweep17(:,4).*1e3;

plot(t,y1,'*','color',[0 0.4470 0.7410])
hold on
plot(t,y2,'*','color',[0.8500    0.3250    0.0980])
plot(t,y2-y1,'color',[0.9290 0.6940 0.1250])

tbl = table(t-t(1), y1);

modelfun = @(b,x) b(1) + b(2) * exp(-b(3)*x(:, 1));  
beta0 = [34, -8, 0.004]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'b-', 'LineWidth', 1);
grid on;
tau(1,3)=1/coefficients(3);

tb2 = table(t-t(1), y2);
beta0 = [45, -5, 0.004];
mdl = fitnlm(tb2, modelfun, beta0);
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'r-', 'LineWidth', 1);
grid on;

tau(2,3)=1/coefficients(3);
%%
t=temperaturemeasurementTP0320211013Sweep18(:,2);
y1=temperaturemeasurementTP0320211013Sweep18(:,3).*1e3;
y2=temperaturemeasurementTP0320211013Sweep18(:,4).*1e3;

plot(t,y1,'*','color',[0 0.4470 0.7410])
hold on
plot(t,y2,'*','color',[0.8500    0.3250    0.0980])
plot(t,y2-y1,'color',[0.9290 0.6940 0.1250])

tbl = table(t-t(1), y1);
modelfun = @(b,x) b(1) + b(2) * exp(-b(3)*x(:, 1));  
beta0 = [75, -16, 0.004]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);

coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'b-', 'LineWidth', 1);
grid on;
tau(1,4)=1/coefficients(3);

tb2 = table(t-t(1), y2);
beta0 = [75, -16, 0.004];
mdl = fitnlm(tb2, modelfun, beta0);

coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'r-', 'LineWidth', 1);
grid on;

tau(2,4)=1/coefficients(3);
%%
t=temperaturemeasurementTP0320211013Sweep19(:,2);
y1=temperaturemeasurementTP0320211013Sweep19(:,3).*1e3;
y2=temperaturemeasurementTP0320211013Sweep19(:,4).*1e3;

plot(t,y1,'*','color',[0 0.4470 0.7410])
hold on
plot(t,y2,'*','color',[0.8500    0.3250    0.0980])
plot(t,y2-y1,'color',[0.9290 0.6940 0.1250])


tbl = table(t-t(1), y1);

modelfun = @(b,x) b(1) + b(2) * exp(-b(3)*x(:, 1));  
beta0 = [75, -16, 0.004]; % Guess values to start with.  Just make your best guess.
mdl = fitnlm(tbl, modelfun, beta0);

coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'b-', 'LineWidth', 1);
grid on;
tau(1,5)=1/coefficients(3);

tb2 = table(t-t(1), y2);
beta0 = [75, -16, 0.004];
mdl = fitnlm(tb2, modelfun, beta0);
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'r-', 'LineWidth', 1);
grid on;

tau(2,5)=1/coefficients(3);
%%
t=temperaturemeasurementTP0320211013Sweep110(:,2);
y1=temperaturemeasurementTP0320211013Sweep110(:,3).*1e3;
y2=temperaturemeasurementTP0320211013Sweep110(:,4).*1e3;

plot(t,y1,'*','color',[0 0.4470 0.7410])
hold on
plot(t,y2,'*','color',[0.8500    0.3250    0.0980])
plot(t,y2-y1,'color',[0.9290 0.6940 0.1250])

tbl = table(t-t(1), y1);

modelfun = @(b,x) b(1) + b(2) * exp(-b(3)*x(:, 1));  
beta0 = [75, -16, 0.004]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.

coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'b-', 'LineWidth', 1);
grid on;
tau(1,6)=1/coefficients(3);

tb2 = table(t-t(1), y2);
beta0 = [75, -16, 0.004];
mdl = fitnlm(tb2, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.
% YAY!!!!

% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'r-', 'LineWidth', 1);
grid on;

tau(2,6)=1/coefficients(3);
%%
t=temperaturemeasurementTP0320211013Sweep112(:,2);
y1=temperaturemeasurementTP0320211013Sweep112(:,3).*1e3;
y2=temperaturemeasurementTP0320211013Sweep112(:,4).*1e3;

plot(t,y1,'*','color',[0 0.4470 0.7410])
hold on
plot(t,y2,'*','color',[0.8500    0.3250    0.0980])
plot(t,y2-y1,'color',[0.9290 0.6940 0.1250])

% Convert X and Y into a table, which is the form fitnlm() likes the input data to be in.
tbl = table(t-t(1), y1);
% Define the model as Y = a + exp(-b*x)
% Note how this "x" of modelfun is related to big X and big Y.
% x((:, 1) is actually X and x(:, 2) is actually Y - the first and second columns of the table.
modelfun = @(b,x) b(1) + b(2) * exp(-b(3)*x(:, 1));  
beta0 = [10, 16, 0.004]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.
% YAY!!!!

% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'b-', 'LineWidth', 1);
grid on;
tau(1,7)=1/coefficients(3);

tb2 = table(t-t(1), y2);
beta0 = [10, 16, 0.004];
mdl = fitnlm(tb2, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.
% YAY!!!!

% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * exp(-coefficients(3)*(t-t(1)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'r-', 'LineWidth', 1);
grid on;

tau(2,7)=1/coefficients(3);
%% Device Characterization 

%%% Load the data 
load('pinchoffdata.mat')

%%% 
figure(2)
t=datafile20211013pinchoffpinBG365(:,1)./1e3;

y=-datafile20211013pinchoffpinBG365(:,4)./5./(2*(1.6e-19)^2/1.6261e-34);

plot(t,y,'color',[0 0.4470 0.7410])
hold on

% Convert X and Y into a table, which is the form fitnlm() likes the input data to be in.
tbl = table(t(96:length(t)), y(96:length(y)));
% Define the model as Y = a + exp(-b*x)
modelfun = @(b,x) 1./(b(1) + b(2) * 1./(x(:, 1)-b(3)));  
beta0 = [1./0.05, 0.3, -1.6]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = 1./(coefficients(1) + coefficients(2) .* 1./(t(96:length(t))-coefficients(3)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t(96:length(t)), yFitted, 'b-', 'LineWidth', 1);
grid on;

%%
figure(2)
t=datafile20211013pinchoffpinBG366(:,1)./1e3;

y=-datafile20211013pinchoffpinBG366(:,4)./5./(2*(1.6e-19)^2/1.6261e-34);

plot(t,y,'color',[0.8500    0.3250    0.0980])
hold on;

% Convert X and Y into a table, which is the form fitnlm() likes the input data to be in.
tbl = table(t(96:length(t)), y(96:length(y)));
% Define the model as Y = a + exp(-b*x)
% Note how this "x" of modelfun is related to big X and big Y.
% x((:, 1) is actually X and x(:, 2) is actually Y - the first and second columns of the table.
modelfun = @(b,x) 1./(b(1) + b(2) * 1./(x(:, 1)-b(3)));  
beta0 = [1./0.05, 0.3, -1.6]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.
% YAY!!!!

% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = 1./(coefficients(1) + coefficients(2) .* 1./(t(96:length(t))-coefficients(3)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t(96:length(t)), yFitted, 'r-', 'LineWidth', 1);
xlabel('V_g [volt]');
ylabel('G(2e^2/h)');
legend('Pinch off trace','Pinch off trace','Fit pinch off trace','Fit pinch off trace');

grid on;

%%
figure(3)
t=datafile20211013pinchoffpinBG367(:,1)./1e3;

y=-datafile20211013pinchoffpinBG367(:,4)./5./(2*(1.6e-19)^2/1.6261e-34);

plot(t,y,'color',[0.9290 0.6940 0.1250])
hold on

% Convert X and Y into a table, which is the form fitnlm() likes the input data to be in.
tbl = table(t(96:length(t)), y(96:length(y)));
% Define the model as Y = a + exp(-b*x)
% Note how this "x" of modelfun is related to big X and big Y.
% x((:, 1) is actually X and x(:, 2) is actually Y - the first and second columns of the table.
modelfun = @(b,x) 1./(b(1) + b(2) * 1./(x(:, 1)-b(3)));  
beta0 = [1./0.05, 0.3, -1.6]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.
% YAY!!!!

% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = 1./(coefficients(1) + coefficients(2) .* 1./(t(96:length(t))-coefficients(3)));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t(96:length(t)), yFitted, 'y-', 'LineWidth', 1);
xlabel('V_g [volt]');
ylabel('G(2e^2/h)');
legend('Pinch off trace','Fit pinch off trace');
grid on;

