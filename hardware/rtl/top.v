module top #(
    parameter integer NUM_INPUTS  = 1312,
    parameter integer NUM_OUTPUTS = 32,
    parameter integer CHAIN_DEPTH = 32,
    parameter integer USE_RELU = 1,
    parameter integer ACC_WIDTH = 24,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE    = "bias.mem"
) (
    input  wire clk,
    input  wire rst_n,
    input  signed [7:0] in_data  [0:NUM_INPUTS-1],
    output reg signed [7:0] out_data [0:NUM_OUTPUTS-1]
);
    reg  signed [7:0] in_data_q   [0:NUM_INPUTS-1];
    wire signed [7:0] neuron_out  [0:NUM_OUTPUTS-1];

    integer i;
    integer j;
    genvar go;

    generate
        for (go = 0; go < NUM_OUTPUTS; go = go + 1) begin : g_neurons
            neuron #(
                .NUM_INPUTS(NUM_INPUTS),
                .NUM_OUTPUTS(NUM_OUTPUTS),
                .OUTPUT_INDEX(go),
                .CHAIN_DEPTH(CHAIN_DEPTH),
                .USE_RELU(USE_RELU),
                .ACC_WIDTH(ACC_WIDTH),
                .WEIGHTS_FILE(WEIGHTS_FILE),
                .BIAS_FILE(BIAS_FILE)
            ) u_neuron (
                .in_data(in_data_q),
                .out_data(neuron_out[go])
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                in_data_q[i] <= 8'sd0;
            end
            for (j = 0; j < NUM_OUTPUTS; j = j + 1) begin
                out_data[j] <= 8'sd0;
            end
        end else begin
            for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                in_data_q[i] <= in_data[i];
            end
            for (j = 0; j < NUM_OUTPUTS; j = j + 1) begin
                out_data[j] <= neuron_out[j];
            end
        end
    end
endmodule
