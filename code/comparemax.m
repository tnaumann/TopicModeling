function mc = comparemax(A, B)

% Custom reduction function for 3-element vector input
if (B(1) > A(1) || ((B(1) == A(1)) && (B(2) < A(2)) && (B(3) == A(3))))
    mc = B;                 % Return the vector with the larger accuracy
else
    mc = A;
end
