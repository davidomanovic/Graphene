%% Generate a Large Perfect Triangular Lattice and Write a LAMMPS .dat File
clear; clc; close all;

%% User Parameters
a = 1.42;       % Lattice constant (you can change this as needed)
n1 = 50;        % Number of repeats along the a1 direction
n2 = 50;        % Number of repeats along the a2 direction

%% Define Triangular Lattice Primitive Vectors
% For a triangular lattice, a common choice is:
%   a1 = [ a,  0 ]
%   a2 = [ a/2, a*sqrt(3)/2 ]
a1 = [a, 0];
a2 = [a/2, a*sqrt(3)/2];

%% Generate Lattice Points
N = n1 * n2;  % Total number of atoms
positions = zeros(N, 2);  % Preallocate for x,y positions

index = 1;
for i = 0:(n1-1)
    for j = 0:(n2-1)
        positions(index,:) = i * a1 + j * a2;
        index = index + 1;
    end
end

%% Determine Simulation Box Bounds
% Use the minimal rectangle that encloses all points.
x_min = min(positions(:,1));
x_max = max(positions(:,1));
y_min = min(positions(:,2));
y_max = max(positions(:,2));

% Optional: add a little buffer if desired.
buffer = 0.0;
xlo = x_min - buffer;
xhi = x_max + buffer;
ylo = y_min - buffer;
yhi = y_max + buffer;

% For a 2D simulation, we assign a small z-range.
zlo = -0.5;
zhi =  0.5;

%% Plot the Lattice to Verify
figure;
plot(positions(:,1), positions(:,2), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 4);
xlabel('x');
ylabel('y');
title('Large Perfect Triangular Lattice');
axis equal;
grid on;

%% Write LAMMPS .dat File
% The file will have the standard LAMMPS data format:
%   Header, number of atoms, number of atom types,
%   Simulation box dimensions, Masses section, and Atoms section.
filename = './triangular.dat';
fid = fopen(filename, 'w');

fprintf(fid, '# LAMMPS data file: Large Triangular Lattice\n\n');
fprintf(fid, '%d atoms\n', N);
fprintf(fid, '1 atom types\n\n');

fprintf(fid, '%.6f %.6f xlo xhi\n', xlo, xhi);
fprintf(fid, '%.6f %.6f ylo yhi\n', ylo, yhi);
fprintf(fid, '%.6f %.6f zlo zhi\n\n', zlo, zhi);

fprintf(fid, 'Masses\n\n');
fprintf(fid, '1 12.0107\n\n');  % Mass for carbon (example)

fprintf(fid, 'Atoms\n\n');
% Writing one atom per line: id, atom type, x, y, z.
for i = 1:N
    fprintf(fid, '%d 1 %.6f %.6f 0.0\n', i, positions(i,1), positions(i,2));
end

fclose(fid);
fprintf('Wrote %s with %d atoms.\n', filename, N);
