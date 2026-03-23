module \2x2_multi_add #(
    parameter integer ACC_WIDTH = 24
) (
    input  signed [7:0]               in0,
    input  signed [7:0]               in1,
    input  signed [7:0]               w0,
    input  signed [7:0]               w1,
    input  signed [ACC_WIDTH-1:0]     accum_in,
    output signed [ACC_WIDTH-1:0]     accum_out
);
    wire signed [15:0] p0;
    wire signed [15:0] p1;
    wire signed [ACC_WIDTH-1:0] p0_ext;
    wire signed [ACC_WIDTH-1:0] p1_ext;

    assign p0 = in0 * w0;
    assign p1 = in1 * w1;
    assign p0_ext = {{(ACC_WIDTH-16){p0[15]}}, p0};
    assign p1_ext = {{(ACC_WIDTH-16){p1[15]}}, p1};
    assign accum_out = accum_in + p0_ext + p1_ext;
endmodule
