%% Parameter
n = 100; %number of field stages

%% Initialize linear programm
c = -ones(1,n);

temp = 1:n-1;
temp = -1./(temp+2);
temp = [eye(n-1) zeros(n-1,1)]+[zeros(n-1,1) diag(temp)];
A = [-ones(n-1,1) eye(n-1); temp; zeros(1,n-1) 1];

temp = 2:n;
temp = -(temp-1).^2;
b = [temp'; zeros(n,1)];

%% Solve linear programm
values = linprog(c,A,b,[],[],[],[]);

actions = strings(n, 1);

for i=1:n
    if i<n
        harvest_ex_value = -(i-1)^2+values(1);
        wait_ex_value = 1/(i+2)*values(i+1);
    else
        harvest_ex_value = -(i-1)^2+values(1);
        wait_ex_value = 1/(i+2)*values(i);
    end
    if harvest_ex_value < wait_ex_value
        actions(i) = "harvest";
    else
        actions(i) = "wait";
    end
    if abs(harvest_ex_value - wait_ex_value) < 0.00001
        actions(i) = "wait or harvest";
    end
end

disp([(1:n)' values actions]);