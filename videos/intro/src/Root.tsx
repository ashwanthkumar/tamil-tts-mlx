import {Composition} from 'remotion';
import {Intro} from './Intro';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Intro"
      component={Intro}
      durationInFrames={300}
      fps={30}
      width={1280}
      height={720}
    />
  );
};
