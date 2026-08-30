#include "EverestOperatorPawn.h"
#include "EverestWorldManager.h"
#include "EverestRobotActor.h"
#include "GameFramework/SpringArmComponent.h"
#include "Camera/CameraComponent.h"
#include "Components/InputComponent.h"
#include "EngineUtils.h"
AEverestOperatorPawn::AEverestOperatorPawn(){PrimaryActorTick.bCanEverTick=true;Root=CreateDefaultSubobject<USceneComponent>(TEXT("Root"));SetRootComponent(Root);Arm=CreateDefaultSubobject<USpringArmComponent>(TEXT("Arm"));Arm->SetupAttachment(Root);Arm->TargetArmLength=430;Arm->bDoCollisionTest=false;Camera=CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));Camera->SetupAttachment(Arm);AutoPossessPlayer=EAutoReceiveInput::Player0;}
void AEverestOperatorPawn::BeginPlay(){Super::BeginPlay();for(TActorIterator<AEverestWorldManager>It(GetWorld());It;++It){Manager=*It;break;}}
void AEverestOperatorPawn::SetupPlayerInputComponent(UInputComponent* I){Super::SetupPlayerInputComponent(I);I->BindAxis(TEXT("RobotForward"),this,&AEverestOperatorPawn::SetForward);I->BindAxis(TEXT("RobotLateral"),this,&AEverestOperatorPawn::SetLateral);I->BindAxis(TEXT("RobotYaw"),this,&AEverestOperatorPawn::SetYaw);I->BindAxis(TEXT("CameraYaw"),this,&AEverestOperatorPawn::CameraYaw);I->BindAxis(TEXT("CameraPitch"),this,&AEverestOperatorPawn::CameraPitch);I->BindAxis(TEXT("CameraZoom"),this,&AEverestOperatorPawn::CameraZoom);I->BindAction(TEXT("TogglePause"),IE_Pressed,this,&AEverestOperatorPawn::TogglePause);}
void AEverestOperatorPawn::CameraYaw(float V){OrbitYaw+=V*1.5f;}void AEverestOperatorPawn::CameraPitch(float V){OrbitPitch=FMath::Clamp(OrbitPitch+V*1.2f,-70.f,10.f);}void AEverestOperatorPawn::CameraZoom(float V){Arm->TargetArmLength=FMath::Clamp(Arm->TargetArmLength-V*45.f,160.f,1200.f);}void AEverestOperatorPawn::TogglePause(){bPaused=!bPaused;if(Manager)Manager->SetSimulationPaused(bPaused);}
void AEverestOperatorPawn::Tick(float Dt){Super::Tick(Dt);Arm->SetRelativeRotation(FRotator(OrbitPitch,OrbitYaw,0));if(Manager&&Manager->GetRobot()){const FVector Target=Manager->GetRobot()->GetPelvisLocation()+FVector(0,0,90);SetActorLocation(FMath::VInterpTo(GetActorLocation(),Target,Dt,5));SendAccumulator+=Dt;if(SendAccumulator>=0.05f){SendAccumulator=0;Manager->SendRobotCommand(Forward*0.45f,Lateral*0.25f,YawRate*0.8f);}}}
