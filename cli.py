import argparse,os
from proposed import *
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['embed','extract']);p.add_argument('--image',required=True);p.add_argument('--watermark');p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);p.add_argument('--record',default='asset-001');p.add_argument('--size',type=int,choices=[64,128,256],default=64);a=p.parse_args()
    key=bytes.fromhex(os.environ['WATERMARK_KEY_HEX']);im=np.array(Image.open(a.image).convert('RGB'))
    if im.shape!=(512,512,3):p.error('512x512 RGB image required')
    if a.operation=='embed':
        if not a.watermark:p.error('--watermark required')
        w=np.asarray(gray_watermark(Image.open(a.watermark)).resize((a.size,a.size),Image.Resampling.BICUBIC),float)
        out,m,_,_=embed_image(im,w,key,a.record,Config(ALPHA[a.size]));Image.fromarray(out).save(a.output);save_meta(a.metadata,m)
    else:
        w,ok,angle=recover(im,load_meta(a.metadata),key);print('authenticated:',ok,'angle:',angle)
        if not ok:raise SystemExit(2)
        Image.fromarray(np.rint(np.clip(w,0,255)).astype(np.uint8)).save(a.output)
if __name__=='__main__':main()
