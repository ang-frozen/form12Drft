import torch
import torch.nn as nn
#---------------------------------------------------------------------------------------------------------
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool across sequence dimension
        self.fc1 = nn.Linear(channels, channels 
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, seq_len = x.shape
        y = self.global_avg_pool(x).view(b, c)  # Squeeze: (b, c, 1) -> (b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)  # Expand back to (b, c, 1)
        return x * y.expand_as(x)  # Channel-wise recalibration

# Example usage
x = torch.randn(32, 64, 100)  # (batch=32, channels=64, sequence_length=100)
se_block = SEBlock1D(channels=64)
output = se_block(x)

print(output.shape)  # Should be (32, 64, 100)
#---------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, ):
        super(ResidualBlock1D, self).__init__()
        
        padding = kernel_size // 2  # Keep output size same
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        
        
    
    def forward(self, x):
        identity = x  # Store original input for residual connection
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
    
        out += identity  # Residual connection
        return self.relu(out)  # Final activation

# Example usage
x = torch.randn(32, 2, 3000)  # (batch=32, channels=64, sequence_length=100)
res_block = ResidualBlock1D(channels=2, kernel_size=3)
output = res_block(x)

print(output.shape)  # Should be (32, 64, 100)

#---------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class ChannelwiseTransformerBlock(nn.Module):
    def __init__(self, channels, seq_len, chunk_size=10, embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
        super(ChannelwiseTransformerBlock, self).__init__()
        
        self.channels = channels
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.num_chunks = seq_len // chunk_size
        
        self.embedding = nn.Linear(chunk_size, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, c, s = x.shape
        assert s == self.seq_len
        
        x_chunks = x.unfold(dimension=2, size=self.chunk_size, step=self.chunk_size)
        x_chunks = x_chunks.permute(0, 2, 1, 3).contiguous()
        
        x_chunks = self.embedding(x_chunks)
        
        x_attended = []
        for i in range(self.channels):
            chunk = x_chunks[:, :, i, :]
            attn_output, _ = self.attention(chunk, chunk, chunk)
            attn_output = self.norm1(attn_output + chunk)
            
            ffn_output = self.ffn(attn_output)
            ffn_output = self.norm2(ffn_output + attn_output)
            x_attended.append(ffn_output)
        
        x_transformed = torch.stack(x_attended, dim=2)
        x_transformed = x_transformed.view(b, c, -1)
        
        return x_transformed

x = torch.randn(32, 2, 3000)
transformer_block = ChannelwiseTransformerBlock(channels=2, seq_len=3000, chunk_size=100, embed_dim=64)
output = transformer_block(x)

print(output.shape)

#---------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_k=1):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size,padding='same'),
                                    nn.Conv1d(in_channels=out_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size,padding='same'),
                                             nn.BatchNorm1d(out_channels),
                                             nn.ELU(),
                                             )
        self.pool = nn.AvgPool1d(kernel_size=pool_k)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        return x

# Example usage
x = torch.randn(32, 2, 3000)  # (batch_size, channels, height, width)
conv_block = DoubleConvBlock(in_channels=2, out_channels=64)
output = conv_block(x)

print(output.shape)  # Expected shape: (32, 64, 64, 64)


#---------------------------------------------------------------------------------------------------------

class SLEEP(nn.Module):
    def __init__(self):
        super(SLEEP, self).__init__()
        self.dc1 = DoubleConvBlock(in_channels=2, out_channels=32, kernel_size=5, pool_k=1)
        self.dc2 = DoubleConvBlock(in_channels=32, out_channels=32, kernel_size=5, pool_k=4)
        self.res1 = ResidualBlock1D(channels=32, kernel_size=3)

        self.dc3 = DoubleConvBlock(in_channels=32, out_channels=64, kernel_size=5, pool_k=1)
        self.dc4 = DoubleConvBlock(in_channels=64, out_channels=64, kernel_size=5, pool_k=4)
        self.res2 = ResidualBlock1D(channels=64, kernel_size=3)

        self.dc5 = DoubleConvBlock(in_channels=64, out_channels=128, kernel_size=5, pool_k=1)
        self.dc6 = DoubleConvBlock(in_channels=128, out_channels=128, kernel_size=5, pool_k=4)
        self.res3 = ResidualBlock1D(channels=128, kernel_size=3)

        self.dc7 = DoubleConvBlock(in_channels=128, out_channels=256, kernel_size=5, pool_k=1)
        self.dc8 = DoubleConvBlock(in_channels=256, out_channels=256, kernel_size=5, pool_k=4)
        self.res4 = ResidualBlock1D(channels=256, kernel_size=3)

        self.at = ChannelwiseTransformerBlock(channels=2, seq_len=3000, chunk_size=100)
        self.sq1 = DoubleConvBlock(in_channels=2, out_channels=128, kernel_size=7)
        self.sqe1 = SEBlock1D(channels=128, reduction=32)
        self.avg1 = nn.AvgPool1d(4)
        self.at2 = ChannelwiseTransformerBlock(channels=128, seq_len=480, chunk_size=100)
        self.at_lin = nn.Linear(8192,5)

        self.res_lin = nn.Linear(2816,5)

        self.final_lin = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=1))
        


    def forward(self, x):
        x1 = self.dc1(x)
        x1 = self.dc2(x1)
        x1 = self.res1(x1)

        x1 = self.dc3(x1)
        x1 = self.dc4(x1)
        x1 = self.res2(x1)

        x1 = self.dc5(x1)
        x1 = self.dc6(x1)
        x1 = self.res3(x1)

        x1 = self.dc7(x1)
        x1 = self.dc8(x1)
        x1 = self.res4(x1)
        x1 = self.res_lin(torch.flatten(x1, 1))

        xa = self.at(x)
        a1 = xa
        xa = self.sq1(xa)
        
        xa = self.sqe1(xa)
        xa = self.avg1(xa)
        xa = self.at2(xa)

        xa = self.avg1(xa)
        a2 = xa
        xa = self.at_lin(torch.flatten(xa, 1))

        out = self.final_lin(torch.cat([x1, xa], 1))

        return out, [x1, xa], a1

# Example usage
x = torch.randn(8, 2, 3000)  # (batch_size, channels, height, width)
model = SLEEP()
output = model(x)

print(output[1][0].shape)  # Expected shape: (32, 64, 64, 64)
