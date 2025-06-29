import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, Platform, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Camera, RotateCcw, Zap, ZapOff, Monitor, Smartphone, CircleCheck as CheckCircle } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useScans } from '@/hooks/useScans';

const { width } = Dimensions.get('window');
const isWeb = Platform.OS === 'web';
const isLargeScreen = width > 768;

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [flash, setFlash] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureSuccess, setCaptureSuccess] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const router = useRouter();
  const { addScan } = useScans();

  const triggerHaptic = () => {
    if (Platform.OS !== 'web') {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    } else {
      // Web alternative - visual feedback could be added here
      console.log('Haptic feedback triggered (web)');
    }
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <Text style={[styles.loadingText, isLargeScreen && styles.loadingTextLarge]}>
            Loading camera...
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={[styles.permissionContainer, isLargeScreen && styles.permissionContainerLarge]}>
          <Camera size={isLargeScreen ? 80 : 64} color="#0066CC" strokeWidth={2} />
          <Text style={[styles.permissionTitle, isLargeScreen && styles.permissionTitleLarge]}>
            Camera Access Required
          </Text>
          <Text style={[styles.permissionText, isLargeScreen && styles.permissionTextLarge]}>
            We need access to your {isWeb ? 'webcam' : 'camera'} to scan skin lesions and moles for analysis.
            {isWeb ? ' Make sure to allow camera access in your browser.' : ''}
          </Text>
          <TouchableOpacity 
            style={[styles.permissionButton, isLargeScreen && styles.permissionButtonLarge]} 
            onPress={requestPermission}>
            <Text style={[styles.permissionButtonText, isLargeScreen && styles.permissionButtonTextLarge]}>
              Grant Permission
            </Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const toggleCameraFacing = () => {
    triggerHaptic();
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  const toggleFlash = () => {
    triggerHaptic();
    setFlash(!flash);
  };

  const capturePhoto = async () => {
    if (isCapturing || !cameraRef.current) return;
    
    try {
      setIsCapturing(true);
      setCaptureSuccess(false);
      triggerHaptic();
      
      console.log('Starting photo capture...');
      
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
        skipProcessing: false,
      });
      
      console.log('Photo captured:', photo);
      
      if (photo && photo.uri) {
        console.log('Adding scan with URI:', photo.uri);
        
        // Add the captured photo to scans
        const newScan = await addScan({
          uri: photo.uri,
          result: 'analyzing',
          location: 'Captured with camera'
        });
        
        console.log('Scan added successfully:', newScan);

        // Show success feedback
        setCaptureSuccess(true);
        
        // Auto-hide success message and show alert
        setTimeout(() => {
          setCaptureSuccess(false);
          Alert.alert(
            'Photo Captured Successfully!',
            `Image captured and saved! ${isWeb ? 'Enhanced desktop analysis' : 'Mobile analysis'} is now processing your scan.`,
            [
              {
                text: 'View My Scans',
                onPress: () => router.push('/data'),
              },
              {
                text: 'Take Another',
                style: 'cancel'
              }
            ]
          );
        }, 1500);
        
      } else {
        console.error('Photo capture failed - no URI returned');
        Alert.alert('Error', 'Failed to capture photo. No image data received.');
      }
    } catch (error) {
      console.error('Camera capture error:', error);
      Alert.alert(
        'Capture Failed', 
        `Failed to capture photo: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`
      );
    } finally {
      setIsCapturing(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={[styles.header, isLargeScreen && styles.headerLarge]}>
        <View style={styles.headerContent}>
          <Text style={[styles.title, isLargeScreen && styles.titleLarge]}>
            Skin Scan {isWeb ? 'Webcam' : 'Camera'}
          </Text>
          <Text style={[styles.subtitle, isLargeScreen && styles.subtitleLarge]}>
            Position the lesion in the center
            {isWeb ? ' - Use your webcam for high-quality analysis' : ''}
          </Text>
        </View>
        {isLargeScreen && (
          <View style={styles.platformIndicator}>
            {isWeb ? <Monitor size={24} color="#0066CC" /> : <Smartphone size={24} color="#0066CC" />}
          </View>
        )}
      </View>

      {/* Camera */}
      <View style={[styles.cameraContainer, isLargeScreen && styles.cameraContainerLarge]}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing={facing}
          flash={flash ? 'on' : 'off'}>
          
          {/* Overlay */}
          <View style={styles.overlay}>
            <View style={[styles.scanArea, isLargeScreen && styles.scanAreaLarge]}>
              <View style={[styles.corner, styles.topLeft]} />
              <View style={[styles.corner, styles.topRight]} />
              <View style={[styles.corner, styles.bottomLeft]} />
              <View style={[styles.corner, styles.bottomRight]} />
            </View>
          </View>

          {/* Controls */}
          <View style={[styles.controls, isLargeScreen && styles.controlsLarge]}>
            <TouchableOpacity 
              style={[styles.controlButton, isLargeScreen && styles.controlButtonLarge]} 
              onPress={toggleFlash}>
              {flash ? (
                <Zap size={isLargeScreen ? 28 : 24} color="#FFFFFF" strokeWidth={2} />
              ) : (
                <ZapOff size={isLargeScreen ? 28 : 24} color="#FFFFFF" strokeWidth={2} />
              )}
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.captureButton, 
                isLargeScreen && styles.captureButtonLarge,
                isCapturing && styles.captureButtonDisabled
              ]}
              onPress={capturePhoto}
              disabled={isCapturing}>
              <View style={[styles.captureButtonInner, isLargeScreen && styles.captureButtonInnerLarge]} />
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.controlButton, isLargeScreen && styles.controlButtonLarge]} 
              onPress={toggleCameraFacing}>
              <RotateCcw size={isLargeScreen ? 28 : 24} color="#FFFFFF" strokeWidth={2} />
            </TouchableOpacity>
          </View>
        </CameraView>
      </View>

      {/* Capture Status Overlay */}
      {isCapturing && (
        <View style={[styles.captureStatus, isLargeScreen && styles.captureStatusLarge]}>
          <View style={styles.captureStatusContent}>
            <Text style={[styles.captureStatusText, isLargeScreen && styles.captureStatusTextLarge]}>
              📸 Capturing photo...
            </Text>
          </View>
        </View>
      )}

      {/* Success Status Overlay */}
      {captureSuccess && (
        <View style={[styles.successStatus, isLargeScreen && styles.successStatusLarge]}>
          <View style={styles.successStatusContent}>
            <CheckCircle size={isLargeScreen ? 48 : 36} color="#10B981" strokeWidth={2} />
            <Text style={[styles.successStatusText, isLargeScreen && styles.successStatusTextLarge]}>
              Photo Captured!
            </Text>
            <Text style={[styles.successStatusSubtext, isLargeScreen && styles.successStatusSubtextLarge]}>
              Adding to your scans...
            </Text>
          </View>
        </View>
      )}

      {/* Instructions */}
      <View style={[styles.instructions, isLargeScreen && styles.instructionsLarge]}>
        <Text style={[styles.instructionTitle, isLargeScreen && styles.instructionTitleLarge]}>
          Scanning Tips
        </Text>
        <View style={isLargeScreen ? styles.instructionGrid : styles.instructionList}>
          <Text style={[styles.instructionText, isLargeScreen && styles.instructionTextLarge]}>
            • Ensure good lighting
          </Text>
          <Text style={[styles.instructionText, isLargeScreen && styles.instructionTextLarge]}>
            • Hold {isWeb ? 'position' : 'camera'} steady
          </Text>
          <Text style={[styles.instructionText, isLargeScreen && styles.instructionTextLarge]}>
            • Position lesion in center frame
          </Text>
          <Text style={[styles.instructionText, isLargeScreen && styles.instructionTextLarge]}>
            • Keep 6-8 inches away
          </Text>
          {isWeb && (
            <Text style={[styles.instructionText, isLargeScreen && styles.instructionTextLarge]}>
              • Allow browser camera access
            </Text>
          )}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#FFFFFF',
  },
  loadingTextLarge: {
    fontSize: 20,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#F9FAFB',
  },
  permissionContainerLarge: {
    padding: 60,
    maxWidth: 600,
    alignSelf: 'center',
  },
  permissionTitle: {
    fontSize: 24,
    fontFamily: 'Inter-Bold',
    color: '#111827',
    marginTop: 16,
    marginBottom: 8,
    textAlign: 'center',
  },
  permissionTitleLarge: {
    fontSize: 32,
    marginTop: 24,
    marginBottom: 16,
  },
  permissionText: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 24,
  },
  permissionTextLarge: {
    fontSize: 18,
    lineHeight: 28,
    marginBottom: 32,
  },
  permissionButton: {
    backgroundColor: '#0066CC',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  permissionButtonLarge: {
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  permissionButtonText: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#FFFFFF',
  },
  permissionButtonTextLarge: {
    fontSize: 18,
  },
  header: {
    padding: 24,
    paddingBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLarge: {
    padding: 40,
    paddingBottom: 24,
  },
  headerContent: {
    flex: 1,
  },
  platformIndicator: {
    marginLeft: 16,
  },
  title: {
    fontSize: 24,
    fontFamily: 'Inter-Bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  titleLarge: {
    fontSize: 32,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#D1D5DB',
  },
  subtitleLarge: {
    fontSize: 18,
  },
  cameraContainer: {
    flex: 1,
    marginHorizontal: 24,
    borderRadius: 16,
    overflow: 'hidden',
  },
  cameraContainerLarge: {
    marginHorizontal: 40,
    borderRadius: 24,
    maxHeight: 600,
    alignSelf: 'center',
    width: '100%',
    maxWidth: 800,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanArea: {
    width: 200,
    height: 200,
    position: 'relative',
  },
  scanAreaLarge: {
    width: 300,
    height: 300,
  },
  corner: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: '#0066CC',
    borderWidth: 3,
  },
  topLeft: {
    top: 0,
    left: 0,
    borderRightWidth: 0,
    borderBottomWidth: 0,
  },
  topRight: {
    top: 0,
    right: 0,
    borderLeftWidth: 0,
    borderBottomWidth: 0,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderRightWidth: 0,
    borderTopWidth: 0,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    borderLeftWidth: 0,
    borderTopWidth: 0,
  },
  controls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 24,
  },
  controlsLarge: {
    padding: 40,
  },
  controlButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  controlButtonLarge: {
    width: 60,
    height: 60,
    borderRadius: 30,
  },
  captureButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#0066CC',
  },
  captureButtonLarge: {
    width: 90,
    height: 90,
    borderRadius: 45,
    borderWidth: 5,
  },
  captureButtonDisabled: {
    opacity: 0.5,
  },
  captureButtonInner: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#0066CC',
  },
  captureButtonInnerLarge: {
    width: 60,
    height: 60,
    borderRadius: 30,
  },
  captureStatus: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureStatusLarge: {
    // Same styles for large screen
  },
  captureStatusContent: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    paddingHorizontal: 32,
    paddingVertical: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  captureStatusText: {
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
    color: '#FFFFFF',
  },
  captureStatusTextLarge: {
    fontSize: 22,
  },
  successStatus: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  successStatusLarge: {
    // Same styles for large screen
  },
  successStatusContent: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    paddingHorizontal: 40,
    paddingVertical: 32,
    borderRadius: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 8,
  },
  successStatusText: {
    fontSize: 20,
    fontFamily: 'Inter-Bold',
    color: '#10B981',
    marginTop: 12,
    marginBottom: 4,
  },
  successStatusTextLarge: {
    fontSize: 24,
    marginTop: 16,
    marginBottom: 8,
  },
  successStatusSubtext: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
  },
  successStatusSubtextLarge: {
    fontSize: 16,
  },
  instructions: {
    padding: 24,
    backgroundColor: '#1F2937',
  },
  instructionsLarge: {
    padding: 40,
  },
  instructionTitle: {
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
    color: '#FFFFFF',
    marginBottom: 12,
  },
  instructionTitleLarge: {
    fontSize: 24,
    marginBottom: 20,
  },
  instructionList: {
    flexDirection: 'column',
  },
  instructionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  instructionText: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#D1D5DB',
    marginBottom: 6,
  },
  instructionTextLarge: {
    fontSize: 16,
    marginBottom: 8,
    width: '48%',
  },
});