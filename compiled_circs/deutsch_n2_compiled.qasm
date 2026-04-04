OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
u(pi/2,0,pi) q[0];
u(pi,0,pi) q[1];
cx q[1],q[0];
u(pi/2,0,pi) q[0];
u(pi/2,0,pi) q[1];
