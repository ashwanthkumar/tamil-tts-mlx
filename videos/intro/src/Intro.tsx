import {
  AbsoluteFill,
  Audio,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {loadFont} from '@remotion/google-fonts/NotoSansTamil';

const {fontFamily: tamilFont} = loadFont();
const SANS =
  'Inter, -apple-system, "Helvetica Neue", Arial, sans-serif';
const ACCENT = '#5b9dff';

const FadeUp: React.FC<{at: number; children: React.ReactNode; y?: number}> = ({
  at,
  children,
  y = 24,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - at, fps, config: {damping: 200}});
  return (
    <div style={{opacity: s, transform: `translateY(${interpolate(s, [0, 1], [y, 0])}px)`}}>
      {children}
    </div>
  );
};

const Caption: React.FC<{tamil: string; en: string}> = ({tamil, en}) => (
  <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', padding: 80}}>
    <FadeUp at={4}>
      <div
        style={{
          fontFamily: tamilFont,
          fontSize: 76,
          color: 'white',
          textAlign: 'center',
          lineHeight: 1.25,
          textWrap: 'balance',
        }}
      >
        {tamil}
      </div>
    </FadeUp>
    <FadeUp at={12}>
      <div style={{fontFamily: SANS, fontSize: 30, color: '#9fb3c8', marginTop: 28, textAlign: 'center'}}>
        {en}
      </div>
    </FadeUp>
  </AbsoluteFill>
);

export const Intro: React.FC = () => {
  return (
    <AbsoluteFill
      style={{
        background: 'radial-gradient(120% 120% at 50% 0%, #16233a 0%, #0a0f1a 60%, #070a12 100%)',
      }}
    >
      {/* footer / brand */}
      <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 40}}>
        <div style={{fontFamily: SANS, fontSize: 22, color: '#5f7591', letterSpacing: 1}}>
          github.com/ashwanthkumar/tamil-tts-mlx
        </div>
      </AbsoluteFill>

      {/* Scene A: title */}
      <Sequence durationInFrames={90}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
          <FadeUp at={6}>
            <div style={{fontFamily: tamilFont, fontSize: 120, color: 'white', fontWeight: 700}}>
              தமிழ் TTS
            </div>
          </FadeUp>
          <FadeUp at={18}>
            <div style={{fontFamily: SANS, fontSize: 34, color: ACCENT, marginTop: 16}}>
              Tamil text-to-speech
            </div>
          </FadeUp>
          <FadeUp at={30}>
            <div style={{fontFamily: SANS, fontSize: 24, color: '#9fb3c8', marginTop: 18}}>
              trained on Apple Silicon · MLX → ONNX · runs on CPU
            </div>
          </FadeUp>
        </AbsoluteFill>
      </Sequence>

      {/* Scene B: demo 1 + audio */}
      <Sequence from={90} durationInFrames={95}>
        <Caption tamil="வணக்கம், இது தமிழ் பேச்சு தொகுப்பு." en={'"Hello, this is Tamil speech synthesis."'} />
        <Audio src={staticFile('demo1.wav')} />
      </Sequence>

      {/* Scene C: demo 2 + audio */}
      <Sequence from={190} durationInFrames={75}>
        <Caption tamil="எப்படி இருக்கிறீர்கள்?" en={'"How are you?"'} />
        <Audio src={staticFile('demo2.wav')} />
      </Sequence>

      {/* Scene D: outro */}
      <Sequence from={265}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
          <FadeUp at={2}>
            <div style={{fontFamily: SANS, fontSize: 40, color: 'white'}}>Listen. Generate. On-device.</div>
          </FadeUp>
        </AbsoluteFill>
      </Sequence>
    </AbsoluteFill>
  );
};
