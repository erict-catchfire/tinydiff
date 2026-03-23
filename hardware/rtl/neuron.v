module neuron #(
    parameter integer NUM_INPUTS  = 1312,
    parameter integer NUM_OUTPUTS = 32,
    parameter integer OUTPUT_INDEX = 0,
    parameter integer CHAIN_DEPTH = 32,
    parameter integer USE_RELU = 1,
    parameter integer ACC_WIDTH = 24,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE    = "bias.mem"
) (
    input  signed [7:0] in_data [0:NUM_INPUTS-1],
    output reg signed [7:0] out_data
);
    reg  signed [7:0] weights_mem [0:NUM_OUTPUTS*NUM_INPUTS-1];
    reg  signed [7:0] bias_mem    [0:NUM_OUTPUTS-1];

    localparam integer EFFECTIVE_CHAIN_DEPTH = (CHAIN_DEPTH > 0) ? CHAIN_DEPTH : 1;
    localparam integer NUM_CHAINS = (NUM_INPUTS + EFFECTIVE_CHAIN_DEPTH - 1) / EFFECTIVE_CHAIN_DEPTH;

    wire signed [ACC_WIDTH-1:0] chain_accum [0:NUM_CHAINS-1][0:EFFECTIVE_CHAIN_DEPTH];
    wire signed [ACC_WIDTH-1:0] partial_sum [0:NUM_CHAINS-1];
    reg  signed [ACC_WIDTH-1:0] sum_of_chains;
    wire signed [ACC_WIDTH-1:0] sum_with_bias;
    wire signed [ACC_WIDTH-1:0] post_relu_value;
    localparam signed [ACC_WIDTH-1:0] SAT_MAX = {{(ACC_WIDTH-8){1'b0}}, 8'sd127};
    localparam signed [ACC_WIDTH-1:0] SAT_MIN = {{(ACC_WIDTH-8){1'b1}}, -8'sd128};

    integer s;
    genvar gc;
    genvar gd;

    function signed [7:0] saturate8;
        input signed [ACC_WIDTH-1:0] value;
        begin
            if (value > SAT_MAX) begin
                saturate8 = 8'sd127;
            end else if (value < SAT_MIN) begin
                saturate8 = -8'sd128;
            end else begin
                saturate8 = value[7:0];
            end
        end
    endfunction

    initial begin
        $readmemh(WEIGHTS_FILE, weights_mem);
        $readmemh(BIAS_FILE, bias_mem);
    end

    generate
        for (gc = 0; gc < NUM_CHAINS; gc = gc + 1) begin : g_chain
            assign chain_accum[gc][0] = {ACC_WIDTH{1'b0}};

            for (gd = 0; gd < EFFECTIVE_CHAIN_DEPTH; gd = gd + 1) begin : g_depth
                localparam integer FLAT_INDEX = (gc * EFFECTIVE_CHAIN_DEPTH) + gd;
                wire signed [7:0] stage_in_val;
                wire signed [7:0] stage_weight_val;

                if (FLAT_INDEX < NUM_INPUTS) begin : g_valid_stage
                    assign stage_in_val = in_data[FLAT_INDEX];
                    assign stage_weight_val = weights_mem[(OUTPUT_INDEX*NUM_INPUTS) + FLAT_INDEX];
                end else begin : g_pad_stage
                    assign stage_in_val = 8'sd0;
                    assign stage_weight_val = 8'sd0;
                end

                mult_2x1 #(
                    .ACC_WIDTH(ACC_WIDTH)
                ) u_mult_2x1 (
                    .in_val(stage_in_val),
                    .weight_val(stage_weight_val),
                    .accum_in(chain_accum[gc][gd]),
                    .accum_out(chain_accum[gc][gd+1])
                );
            end

            assign partial_sum[gc] = chain_accum[gc][EFFECTIVE_CHAIN_DEPTH];
        end
    endgenerate

    always @(*) begin
        sum_of_chains = {ACC_WIDTH{1'b0}};
        for (s = 0; s < NUM_CHAINS; s = s + 1) begin
            sum_of_chains = sum_of_chains + partial_sum[s];
        end
    end

    mult_2x1 #(
        .ACC_WIDTH(ACC_WIDTH)
    ) u_bias_add (
        .in_val(8'sd1),
        .weight_val(bias_mem[OUTPUT_INDEX]),
        .accum_in(sum_of_chains),
        .accum_out(sum_with_bias)
    );

    assign post_relu_value =
        ((USE_RELU != 0) && (sum_with_bias < {ACC_WIDTH{1'b0}})) ? {ACC_WIDTH{1'b0}} : sum_with_bias;

    always @(*) begin
        out_data = saturate8(post_relu_value);
    end
endmodule
