a  = 1.42;        % lattice param (Å)
nx = 12;          
ny = round(sqrt(3) * nx);

filename = './mac_model.dat';
writeresults = true;
showresults = true;

% Unit Cell Dimensions:
A = 3 * a;
B = sqrt(3) * a;

% Coordinates of the 4 atoms in the unit cell:
base = [ 0.0,   0.0, 0.0;
         a/2,   B/2, 0.0;
         A/2,   B/2, 0.0;
         2*a,   0.0, 0.0 ];

N = size(base,1) * nx * ny; % Total number of atoms

% Calculate the coordinates of the atoms in the layer:
coords = zeros(N, 3); 
id = 0;
for ix = 0:(nx-1)
    for iy = 0:(ny-1)
        for iatom = 1:size(base,1)
            id = id + 1;
            coords(id,:) = base(iatom,:) + [ix * A, iy * B, 0];
        end
    end
end

if showresults
    figure;
    hold on;
    plot(coords(:,1), coords(:,2), 'ob', 'MarkerSize', 6, 'DisplayName', 'Atoms');
    plot(base(:,1), base(:,2), '.r', 'MarkerSize', 20, 'DisplayName', 'Unit Cell Basis');
    axis equal;
    xlabel('x (Å)');
    ylabel('y (Å)');
    title('Graphene Sheet (2D)');
    legend show;
    hold off;
end

if writeresults
    fid = fopen(filename, 'w');
    fprintf(fid, 'Graphene sheet a=%g\n', a);
    fprintf(fid, '%g atoms\n\n', N);
    fprintf(fid, '1 atom types\n\n');
    
    % Write the simulation box dimensions.
    fprintf(fid, '0 %g xlo xhi\n', A * nx);
    fprintf(fid, '0 %g ylo yhi\n', B * ny);
    fprintf(fid, '-10 10 zlo zhi\n\n');
    
    fprintf(fid, 'Masses\n\n');
    fprintf(fid, '1 12.0107\n\n');

    fprintf(fid, 'Atoms\n\n');
    for i = 1:N
        fprintf(fid, '%g 1 %g %g %g\n', i, coords(i,1), coords(i,2), coords(i,3));
    end
    
    fclose(fid);
    fprintf('Graphene supercell with %g atoms written to %s\n', N, filename);
end