module mult_2x1 #(
    parameter integer ACC_WIDTH = 24
) (
    input  signed [7:0]               in_val,
    input  signed [7:0]               weight_val,
    input  signed [ACC_WIDTH-1:0]     accum_in,
    output signed [ACC_WIDTH-1:0]     accum_out
);
    wire signed [15:0] product;
    wire signed [ACC_WIDTH-1:0] product_ext;

    assign product = in_val * weight_val;
    assign product_ext = {{(ACC_WIDTH-16){product[15]}}, product};
    assign accum_out = accum_in + product_ext;
endmodule
